clc;
clear;

n_protein = 274;

fileFolder='2_PSSMs/Test_PSSMs_137P_137N';
dirOutput=dir(fullfile(fileFolder,'*.txt'));
PSSM_XXXX={dirOutput.name}';
PSSM_XXXX = natsortfiles(PSSM_XXXX);
fileNames_PSSM = [];
for i=1:n_protein
	path_way = [fileFolder '/' cell2mat(PSSM_XXXX(i))];
	lujing=cellstr(path_way);
	fileNames_PSSM = [fileNames_PSSM;lujing];
end


%%%%%%%%%%% Features extraction from PSSM %%%%%%%%%%%%%%%% 

for i=1:n_protein
    i
	files_name = cell2mat(fileNames_PSSM(i));

	PSSM_Matrix = Read_Text_files_PSSM(files_name);
    PSSM_IMG = uint8(255 * mat2gray(PSSM_Matrix));

    %%%%%%%%%%% RICLBP-PSSM %%%%%%%%%%%%%%%%
    ricfeat=Process_RIC(PSSM_IMG);
    Insulin_PSSM_RICLBP_274(i,:)=ricfeat;
end

%%%%%%%%%%%%%%%%%%%%%%%% SAVE FILES %%%%%%%%%%%%%%%%%%%%%%%%%
save Insulin_PSSM_RICLBP_274 Insulin_PSSM_RICLBP_274;
csvwrite('Insulin_PSSM_RICLBP_274.csv',Insulin_PSSM_RICLBP_274);
  
