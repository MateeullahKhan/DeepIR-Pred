function RICLBP_feat = Process_RIC(image)
M=getMap();
AC2= [cvtRICLBP(image,1,2,M) cvtRICLBP(image,2,4,M);];
    AC2(find(isnan(AC2)))=0;
    RICLBP_feat=AC2;
end