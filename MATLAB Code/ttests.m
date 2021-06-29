clear;
clc;

thematrix = readmatrix('AUCs.xlsx');
hmat = zeros(15);
pmat = zeros(15);

%Run the T-tests
for a=1:1:15
    bcount = 15;
    
    for b=1:1:bcount
        [h,p] = ttest2(thematrix(:,a), thematrix(:,b));
        hmat(a,b) = h;
        pmat(a,b) = p;
        
    end
    
    bcount = bcount-1;
    
end

%Delete the redundant lower triangles
hmat = triu(hmat);
pmat = triu(pmat);

image(hmat.*500)