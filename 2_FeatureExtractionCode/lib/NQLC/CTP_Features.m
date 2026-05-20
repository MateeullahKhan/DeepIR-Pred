function Matrix=CTP_Features(protein_groupings,groups_no)

L = length(protein_groupings);
Matrix = [];
protein_groupings_index='123';
protein_number_index=1:1:3;
%%%% composition 
CD = zeros(1,groups_no);
for i=1:groups_no
    CD(i)= length(regexp(protein_groupings,protein_groupings_index(i)))/L;
end
Matrix=[Matrix,CD];
%%%% transition 
TD = zeros(1,groups_no*(groups_no-1)/2);

temp = zeros(groups_no,groups_no);
for i=1:L-1
    m=protein_number_index(:,findstr(protein_groupings_index,protein_groupings(i)));
    n=protein_number_index(:,findstr(protein_groupings_index,protein_groupings(i+1)));
    temp(m,n)=temp(m,n)+1;    
end

k=1;
for i=1:groups_no-1
    for j=i+1:groups_no
        TD(k)=(temp(i,j)+temp(j,i))/(L-1);
        k=k+1;
    end
end

Matrix=[Matrix,TD];

%%% distribution 
DD = zeros(1,groups_no*5);
for i=1:groups_no
    j=i*5-4;
    temp=regexp(protein_groupings,protein_groupings_index(i));
    if isempty(temp)
        continue;
    end
    DD(j)=temp(1)/L;
    DD(j+1)=temp(ceil(length(temp)/4))/L;
    DD(j+2)=temp(ceil(length(temp)/2))/L;
    DD(j+3)=temp(ceil(3*length(temp)/4))/L;
    DD(j+4)=temp(end)/L;    
end
Matrix=[Matrix,DD];   
