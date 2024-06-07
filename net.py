import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch import nn
from sklearn.metrics import accuracy_score, f1_score


cls_loss_fn = nn.CrossEntropyLoss()
reg_loss_fn = nn.MSELoss()

reg_metric = nn.L1Loss()
def cls_metric(logit, label):
    pred_label = torch.argmax(logit, dim=-1)
    acc = accuracy_score(pred_label.cpu(), label.cpu())
    f1 = f1_score(pred_label.cpu(), label.view(-1).cpu(), average="macro")
    return acc, f1

# ------------------ LSTM ------------------------
class MultitaskLSTM(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(MultitaskLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, out.shape[2])
        cls_out = self.cls(cls_out)

        out, _ = self.lstm2(out[:, -1, :].view(x.size(0), 1, -1))
        out = self.dropout(out)
        reg_out = self.reg(out[:, -1, :]).flatten()

        return reg_out, cls_out
    

class ClassifyLSTM(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(ClassifyLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)

    def forward(self, x):
        print(x.shape)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, out.shape[2])
        cls_out = self.cls(cls_out)

        return cls_out

class RegressionLSTM(nn.Module):
    
    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(RegressionLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Linear(hidden_size_2, 1)

        
    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out[:, -1, :].view(x.size(0), 1, -1))
        out = self.dropout(out)
        reg_out = self.reg(out[:, -1, :]).flatten()

        return reg_out

# ------------------ MLP ------------------------

class ClassifyMLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, output_size)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.cls(x)
        x = x.view(-1, x.shape[-1])
        return x
    

class ClassifyGRU(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size_1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = self.dropout(out)
        out = out.contiguous().view(-1, out.size(-1))  
        out = self.cls(out)
        return out


class RegressionGRU(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(RegressionGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size_1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out, _ = self.gru2(out)
        out = self.dropout(out)
        out = out.contiguous().view(-1, out.size(-1))  
        out = self.cls(out)
        return out

class MultitaskGRU(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(MultitaskGRU, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size_1, batch_first=True)
        self.gru2 = nn.GRU(hidden_size_1, hidden_size_2, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        cls_out = out[:, -1, :]  # Sử dụng đầu ra của bước cuối cùng cho phân loại
        cls_out = self.cls(cls_out)

        out, _ = self.gru2(out)
        out = self.dropout(out)
        reg_out = out[:, -1, :]  
        reg_out = self.reg(reg_out).squeeze()
        
        return reg_out, cls_out



# ------------------ RNN ------------------------

class ClassifyRNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out.contiguous().view(-1, out.size(-1)) 
        out = self.cls(out)
        return out

class RegressionRNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(RegressionRNN, self).__init__()
        self.rnn1 = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        reg_out = self.reg(out).squeeze()  
        return reg_out

class MultitaskRNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size,
            dropout
        ):
        super(MultitaskRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size_1, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_size_1, output_size)
        self.reg = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        cls_out = out.contiguous()
        cls_out = cls_out.view(-1, cls_out.size(-1)) 
        cls_out = self.cls(cls_out)

        out, _ = self.rnn2(out)
        out = self.dropout(out)
        out = out[:, -1, :]
        reg_out = self.reg(out).squeeze()  

        return reg_out, cls_out
    
    


#------------------ MLSTM-FCN ----------------------

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# Thử 1: Ko ổn lắm
# Thử 2: num_lstm_out=64, lstm_drop_p=0.5 ==> đớ
# Thử cuối: giảm lstm drop out + lại: bỏ softmax
# Thử cuối: tăng 2 drop out lên 0.6 + 0.5 ver3
#Thử cuối tăng tiếp lstm drop out lên 0.8 ver4
# Tăng nốt drop out -> 0.7 0.65
class ClassifyMLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                    num_lstm_out=128, num_lstm_layers=1, 
                    conv1_nf=16, conv2_nf=32, conv3_nf=16,
                    lstm_drop_p=0.7, fc_drop_p=0.65):
        super(ClassifyMLSTMfcn, self).__init__()

        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.fc = nn.Linear(self.conv3_nf+self.num_lstm_out, self.num_classes)
    
    def forward(self, x, seq_lens):
        ''' input x should be in size [B,T,F], where 
            B = Batch size
            T = Time samples
            F = features
        '''
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, 
                                                batch_first=True, 
                                                enforce_sorted=False)
        x1, (ht,ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
                                                    padding_value=0.0)
        x1 = x1[:,-1,:]
        x2 = x.transpose(2,1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2,2)
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.fc(x_all)
        # x_out = F.log_softmax(x_out, dim=1)

        return x_out
    
# Note, sau khi giảm drop out tu 0.7 xuong 0.4, giữ nguyên cấu trúc 128 và 32-64-32, test MAE tang
# Thu tang drop lstm len 0.75, giam num_lstm_out xuong 64
# Thử giảm tiếp độ phức tạp của cnn, num_lstm_out=128, num_lstm_layers=1, conv1_nf=16, conv2_nf=32, conv3_nf=16, lstm_drop_p=0.7, fc_drop_p=0.3 ==> best cmnr khoảng 1.7
# =======================> sẽ thử layout này lại với task classify và implement multitask
# Thử giảm drop out -> 0.5 ==> giảm tiếp test MAE omggggggggggg  => test MAE 1.6 mấy
# Thử tăng câu trúc conv nhân đôi 32-64-32, num_lstm_out=64 ==> tăng lên test MAE: 1.931
# Thử: tăng lại num_lstm_out = 128, num_lstm_layers=2 conv: 16-32-16 ==> giảm xuống 1.7237
# Thử: cấu trúc như trên, nhưng giảm num lstm layers = 1  ver 7  ==> khoảng 1.71 ==> Turn out to be the best
# Thử: tăng cấu trúc conv 32-64-32, giảm lstm drop out xuống 0.45 ver 8 ==> 2.0 ????????
# Kết luận chung, cấu trúc nên là 16-32-16
# Thử từ the best: thử với lstm drop out là 0.4


class RegressionMLSTMfcn(nn.Module):
    def __init__(self, *, max_seq_len, num_features,
                    num_lstm_out=128, num_lstm_layers=1, 
                    conv1_nf=16, conv2_nf=32, conv3_nf=16,
                    lstm_drop_p=0.4, fc_drop_p=0.3):
        super(RegressionMLSTMfcn, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        
        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)
        self.se2 = SELayer(self.conv2_nf)

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        # Điều chỉnh lớp đầu ra để có 1 đơn vị đầu ra
        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, 1)
    
    def forward(self, x, seq_lens):
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        x1, (ht, ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, padding_value=0.0)
        x1 = x1[:, -1, :]
        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)
        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_all)
        x_out = x_out.squeeze(-1)  # Loại bỏ chiều cuối cùng để kích thước từ [48, 1] thành [48]

        return x_out   

# Test 1: Bị đứng nguyên ở các số liệu
# Test 2: ver 2: lstm_drop_p=0.4, fc_drop_p=0.3 ==> lstm_drop_p=0.7, fc_drop_p=0.6
# Test 3: ver 3: tăng lên conv1_nf=32, conv2_nf=64, conv3_nf=32

class MultitaskMLSTMfcn(nn.Module):
    def __init__(self, *, num_classes, max_seq_len, num_features,
                    num_lstm_out=128, num_lstm_layers=1, 
                    conv1_nf=32, conv2_nf=64, conv3_nf=32,
                    lstm_drop_p=0.7, fc_drop_p=0.65):
        super(MultitaskMLSTMfcn, self).__init__()

        # Common attributes
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        # LSTM configuration
        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        # Convolutional layers configuration
        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        # Dropout configuration
        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_features, 
                            hidden_size=num_lstm_out,
                            num_layers=num_lstm_layers,
                            batch_first=True)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(num_features, conv1_nf, 8)
        self.conv2 = nn.Conv1d(conv1_nf, conv2_nf, 5)
        self.conv3 = nn.Conv1d(conv2_nf, conv3_nf, 3)

        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(conv1_nf)
        self.bn2 = nn.BatchNorm1d(conv2_nf)
        self.bn3 = nn.BatchNorm1d(conv3_nf)

        # Squeeze-and-Excitation layers
        self.se1 = SELayer(conv1_nf)
        self.se2 = SELayer(conv2_nf)

        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_drop_p)

        # Fully connected layers for each task
        self.fc_reg = nn.Linear(conv3_nf + num_lstm_out, 1)  # Regression
        self.fc_cls = nn.Linear(conv3_nf + num_lstm_out, num_classes)  # Classification

    def forward(self, x, seq_lens):
        # Pack the sequence for LSTM processing
        x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(x_packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=0.0)
        lstm_out_last = lstm_out[:, -1, :]

        # Convolutional layers processing
        x = x.transpose(2, 1)  # Prepare for Conv1d
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.se1(x)
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.se2(x)
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = torch.mean(x, 2)  # Global Average Pooling

        # Concatenate LSTM and CNN features
        x_all = torch.cat((lstm_out_last, x), dim=1)

        # Classification output
        cls_out = self.fc_cls(x_all)
        # Undo the softmax since it is contained in CrossEntropyLoss
        # cls_out = F.log_softmax(cls_out, dim=1) 

        # Regression output
        reg_out = self.fc_reg(x_all)
        reg_out = reg_out.squeeze(-1)  # Remove last dimension to match the required output shape

        return cls_out, reg_out



# --------------------- CNN ----------------
    
class ClassifyCNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, dropout):
        super(ClassifyCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size_1, out_channels=hidden_size_2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_size_2 * 75, 512)  # Tính toán kích thước đầu ra từ Convolutional Layers
        self.fc2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    

#------------------ TCN ----------------------


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ClassifyTCN(nn.Module):
    def __init__(self, input_size, num_channels, output_size, kernel_size, dropout):
        super(ClassifyTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(1))  # Reshape to [batch_size * 300, 3, 48]
        y = self.tcn(x)  # TCN
        y = y[:, :, -1]  # Take last output for prediction
        y = self.linear(y)
        return y

class MultitaskTCN(nn.Module):
    def __init__(self, input_size, num_channels, output_size_cls, output_size_reg, kernel_size, dropout):
        super(MultitaskTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        # Classifier head
        self.linear_cls = nn.Linear(num_channels[-1], output_size_cls)
        # Regression head
        self.linear_reg = nn.Linear(num_channels[-1], output_size_reg)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        y = self.tcn(x)  
        cls_out = self.linear_cls(y[:, :, -1]) 
        avg_pool = F.avg_pool1d(y, kernel_size=y.size()[2]).squeeze(-1)
        reg_out = self.linear_reg(avg_pool).squeeze(-1)
        
        return cls_out, reg_out



class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class RegressionTCN(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, output_size, dropout):
        super(RegressionTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNLayer(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size)]

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, features, seq_len]
        x = self.tcn(x)
        x = F.avg_pool1d(x, kernel_size=x.size()[2])  # Global average pooling
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.squeeze(-1)  # Squeeze the tensor to remove the last dimension
        return x
    

#------------------ FCN ----------------------

class RegressionFCN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, dropout):
        super(RegressionFCN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)  
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)  
        self.fc3 = nn.Linear(hidden_size_2, 1)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, 3)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(-1, 300) 
        x = torch.mean(x, dim=1)  
        return x
    
    