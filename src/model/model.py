from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
import torch.nn.functional as F
from src.functions.utils import to_list

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentivePooling, self).__init__()
        self.hidden_size = hidden_size
        self.q_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_projection = nn.Linear(self.hidden_size, self.hidden_size)

    def __call__(self, query, context, context_mask):
        context_mask = context_mask.masked_fill(context_mask == 0, -100)
        # query : [batch, hidden]
        # context : [batch, seq, hidden]
        # context_mask : [batch, seq, window]

        q = self.q_projection(query).unsqueeze(-1)
        c = self.c_projection(context)

        # q : [batch, hidden, 1]
        # c : [batch, seq, hidden]

        att = c.bmm(q)
        # att : [batch, seq, 1]

        # masked_att : [batch, window, seq]
        expanded_att = att.expand(-1, -1, 200)
        masked_att = expanded_att + context_mask

        # att_alienment : [batch, window, seq, 1]
        att_alienment = F.softmax(masked_att, dim=1).transpose(1, 2)




        # result : [batch, window, hidden]
        result = att_alienment.bmm(c)

        return result
class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnswering, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.electra = ElectraModel(config)

        # bi-gru layer 선언
        self.bi_gru = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.question_encoder = nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                                       num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        # self.sent_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=1, bias=True)
        # bi-gru layer output을 2의 크기로 줄여주기 위한 fnn

        self.att_pool = AttentivePooling(config.hidden_size)
        self.sent_gru =  nn.GRU(input_size=config.hidden_size, hidden_size=int(config.hidden_size / 2),
                             num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.projection_layer = nn.Linear(config.hidden_size, int(config.hidden_size/2))
        self.qa_outputs = nn.Linear(int(config.hidden_size/2), config.num_labels)


        # ELECTRA weight 초기화
        self.init_weights()
    def _get_question_vector(self, question_mask, sequence_outputs):
        question_mask = question_mask.unsqueeze(-1)

        encoded_question = sequence_outputs * question_mask
        encoded_question = encoded_question[:, :64, :]
        question_gru_outputs, question_gru_states = self.question_encoder(encoded_question)
        question_vector = torch.cat([question_gru_states[0], question_gru_states[1]], -1)
        return question_vector

    def _get_sentence_vector(self, sentence_mask, sequence_outputs, question_vector):
        # one_hot_sent_mask : [batch, 200, 512]
        one_hot_sent_mask = F.one_hot(sentence_mask, 200).float()
        # sent_output : [batch, window, hidden]
        sent_output = self.att_pool(question_vector, sequence_outputs, one_hot_sent_mask)
        return sent_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            sentence_mask=None,
            question_mask=None,
            inputs_embeds=None,
            start_positions = None,
            end_positions = None
    ):
        # outputs : [1, batch_size, seq_length, hidden_size]
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]

        # gru_output : [batch_size, seq_length, gru_hidden_size*2]
        tok_gru_output, _ = self.bi_gru(sequence_output)

        question_vector = self._get_question_vector(question_mask, tok_gru_output)

        sent_output = self._get_sentence_vector(sentence_mask, tok_gru_output, question_vector)

        sent_gru_output, _ = self.sent_gru(sent_output)

        batch_size = tok_gru_output.size(0)
        tok_length = tok_gru_output.size(1)
        sent_length = sent_gru_output.size(1)

        tok_projection_outputs = self.projection_layer(tok_gru_output)
        # [batch, tok_len, hidden/2]

        sent_projection_outputs = self.projection_layer(sent_gru_output)
        # [batch, sent_len, hidden/2]

        norm_vectors = torch.cat([tok_projection_outputs, sent_projection_outputs], 1)
        # [batch, tok_len + sent_len, hidden/2]

        start_expanded_norm_vectors = norm_vectors.unsqueeze(2).expand(-1, -1, tok_length + sent_length, -1)
        end_expanded_norm_vectors = norm_vectors.unsqueeze(1).expand(-1, tok_length + sent_length, -1, -1)

        span_matrix = start_expanded_norm_vectors + end_expanded_norm_vectors
        # [batch, tok_len + sent_len, tok_len + sent_len, hidden/2]

        valid_span_matrix_logits = self.qa_outputs(span_matrix)
        # [batch, tok_len + sent_len, tok_len+sent_len, 2]
        _, span_matrix_logits = valid_span_matrix_logits.split(1, dim=-1)
        sm_logits = span_matrix_logits.squeeze(-1)
        # [batch, tok_len + sent_len, tok_len+sent_len]

        # 학습 시
        if start_positions is not None and end_positions is not None:
            span_matrix_loss_fct = nn.CrossEntropyLoss(reduction='none')
            valid_loss_fct = nn.CrossEntropyLoss()

            # start/end에 대해 loss 계산
            # [batch, num_answers] : [[13, 87, 0, 0, 0], [], ... ]
            # [batch, num_answers] : [[ 1,  3, 0, 0, 0], [], ... ]
            row_labels = torch.zeros([batch_size, tok_length + sent_length], dtype=torch.long).cuda()
            col_labels = torch.zeros([batch_size, tok_length + sent_length], dtype=torch.long).cuda()
            valid_labels = torch.zeros([batch_size, tok_length + sent_length, tok_length + sent_length], dtype=torch.long).cuda()
            for batch_idx in range(batch_size):
                for answer_idx in range(len(start_positions[batch_idx])):
                    if start_positions[batch_idx][answer_idx] == 0:
                        continue
                    row_labels[batch_idx][start_positions[batch_idx][answer_idx]] = end_positions[batch_idx][answer_idx]
                    col_labels[batch_idx][end_positions[batch_idx][answer_idx]] = start_positions[batch_idx][answer_idx]
                    valid_labels[batch_idx][start_positions[batch_idx][answer_idx]][end_positions[batch_idx][answer_idx]] = 1

            row_position_mask = torch.sum(valid_labels, 2).cuda()
            col_position_mask = torch.sum(valid_labels, 1).cuda()

            row_loss = span_matrix_loss_fct(sm_logits.view(-1, tok_length + sent_length), row_labels.view(-1)).reshape(
                batch_size, tok_length + sent_length)
            col_loss = span_matrix_loss_fct(
                sm_logits.transpose(1, 2).reshape(batch_size * (tok_length + sent_length), tok_length + sent_length),
                col_labels.view(-1)).reshape(batch_size, tok_length + sent_length)

            weighted_row_loss = row_loss / torch.sum(row_position_mask, 1, keepdim=True)
            weighted_col_loss = col_loss / torch.sum(col_position_mask, 1, keepdim=True)
            final_row_loss = torch.mean(torch.sum(weighted_row_loss * row_position_mask, 1))
            final_col_loss = torch.mean(torch.sum(weighted_col_loss * col_position_mask, 1))
            ####################################################################################################################################
            # 최종 loss 계산

            matrix_loss = (final_row_loss + final_col_loss) / 2
            valid_loss = valid_loss_fct(span_matrix_logits.view(-1, 2), valid_labels.view(-1,))
            total_loss = matrix_loss + valid_loss

            # outputs : (total_loss, start_logits, end_logits)

            return total_loss, matrix_loss, valid_loss
        valid_score = F.softmax(valid_span_matrix_logits, dim=-1).split(1, dim=-1)[1].squeeze(-1)
        return valid_score, sm_logits
