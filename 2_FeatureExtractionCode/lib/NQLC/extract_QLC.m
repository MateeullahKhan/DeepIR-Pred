clc;
clear all;
feature_QLC=[];

[data, sequence] = fastaread('AIP-Training.txt')
Total_Seq_train=size(sequence,2);

for i=1:(Total_Seq_train)
    i
    SEQ=sequence(i);
	FF=mctd(SEQ);
    SEQ=cell2mat(SEQ);
    feature_QLC(i,:)=FF;
end
AIps_QLC_Training=[feature_QLC];
save AIps_QLC_Training AIps_QLC_Training;
csvwrite('AIps_QLC_Training.csv',AIps_QLC_Training);
