function F = mctd(protein)

F=[];
protein=cell2mat(protein);
%grouping 
Hydro_protein = protein;
Hydro_protein=regexprep(Hydro_protein,'X|B|Z|J|O|U',''); 
Hydro_protein=regexprep(Hydro_protein,'R|K|E|D|Q|N','1');
Hydro_protein=regexprep(Hydro_protein,'G|A|S|T|P|H|Y','2');
Hydro_protein=regexprep(Hydro_protein,'C|L|V|I|M|F|W','3');

Volume_protein = protein;
Volume_protein=regexprep(Volume_protein,'X|B|Z|J|O|U',''); 
Volume_protein=regexprep(Volume_protein,'G|A|S|T|P|D','1');
Volume_protein=regexprep(Volume_protein,'C|N|V|E|Q|I|L','2');
Volume_protein=regexprep(Volume_protein,'M|H|K|F|R|Y|W','3');

Polarity_protein = protein;
Polarity_protein=regexprep(Polarity_protein,'X|B|Z|J|O|U','');
Polarity_protein=regexprep(Polarity_protein,'L|I|F|W|C|M|V|Y','1');
Polarity_protein=regexprep(Polarity_protein,'P|A|T|G|S','2');
Polarity_protein=regexprep(Polarity_protein,'H|Q|R|K|N|E|D','3');

Polarizability_protein = protein;
Polarizability_protein=regexprep(Polarizability_protein,'X|B|Z|J|O|U',''); 
Polarizability_protein=regexprep(Polarizability_protein,'G|A|S|T|D','1');
Polarizability_protein=regexprep(Polarizability_protein,'C|P|N|V|E|Q|I|L','2');
Polarizability_protein=regexprep(Polarizability_protein,'K|M|H|F|R|Y|W','3');

Charge_protein = protein;
Charge_protein=regexprep(Charge_protein,'X|B|Z|J|O|U','');
Charge_protein=regexprep(Charge_protein,'K|R','1');
Charge_protein=regexprep(Charge_protein,'A|N|C|Q|G|H|I|L|M|F|P|S|T|W|Y|V','2');
Charge_protein=regexprep(Charge_protein,'D|E','3');

SecondaryStruct_protein = protein;
SecondaryStruct_protein=regexprep(SecondaryStruct_protein,'X|B|Z|J|O|U','');
SecondaryStruct_protein=regexprep(SecondaryStruct_protein,'E|A|L|M|Q|K|R|H','1');
SecondaryStruct_protein=regexprep(SecondaryStruct_protein,'V|I|Y|C|W|F|T','2');
SecondaryStruct_protein=regexprep(SecondaryStruct_protein,'G|N|P|S|D','3');

SolventAccessiblilict_protein = protein;
SolventAccessiblilict_protein=regexprep(SolventAccessiblilict_protein,'X|B|Z|J|O|U','');
SolventAccessiblilict_protein=regexprep(SolventAccessiblilict_protein,'A|L|F|C|G|I|V|W','1');
SolventAccessiblilict_protein=regexprep(SolventAccessiblilict_protein,'R|K|Q|E|N|D','2');
SolventAccessiblilict_protein=regexprep(SolventAccessiblilict_protein,'M|S|P|T|H|Y','3');

Seq_Arrey = [Hydro_protein;Volume_protein;Polarity_protein;Polarizability_protein;Charge_protein;SecondaryStruct_protein;SolventAccessiblilict_protein];



%CTD

for i=1:size(Seq_Arrey,1)
	region_seq = Seq_Arrey(i,:);
	region_value = CTP_Features(region_seq,3);  %21
	F=[F ,region_value];
end


%20

Length_P = length(protein);
protein_residue_type_index=['A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' 'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'Y'];
residue_type_number_index=1:1:20;
%composition
F_A = zeros(1,20);
for i=1:20
	num_a = length(regexp(protein,protein_residue_type_index(i)));
	F_A(i)= num_a/Length_P;
end

F = [F ,F_A];

F(find(isnan(F)))=0;
F(find(isinf(F)))=0;