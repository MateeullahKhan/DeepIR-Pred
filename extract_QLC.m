clc;
clear all;
feature_QLC=[];

[data, sequence] = fastaread('IR_test_P137_N137.fasta');
Total_Seq_train=size(sequence,2);

for i=1:(Total_Seq_train)
    i
    SEQ=sequence(i);
	FF=mctd(SEQ);
    SEQ=cell2mat(SEQ);
    feature_QLC(i,:)=FF;
end
Insulin_QLC_274=[feature_QLC];
save Insulin_QLC_274 Insulin_QLC_274;
csvwrite('Insulin_QLC_274.csv',Insulin_QLC_274);
