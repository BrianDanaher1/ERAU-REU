clear;
clc;

%Labels
basestrings = '';
run1 = 1;
string = '';

runs = num2str([1:60].','%01d');

for n=1:1:60
    string = [string, 'RUN', ' ', runs(n,1), runs(n,2)];
    basestrings = [basestrings; string];
    string = '';
    
end

basestrings = cellstr(basestrings);

basestring2 = '';
string1 = '';
string2 = '';
string3 = '';
string4 = '';
string5 = '';

stringset2 = {};
networktypes = {' NETWORK 1', ' NETWORK 2', ' NETWORK 3'};

for n=1:1:5
    string1 = [string1; networktypes(1,1); networktypes(1,1); networktypes(1,1); networktypes(1,1)];
    string2 = [string2; networktypes(1,2); networktypes(1,2); networktypes(1,2); networktypes(1,2)];
    string3 = [string3; networktypes(1,3); networktypes(1,3); networktypes(1,3); networktypes(1,3)];
    stringset2 = [stringset2;string1;string2;string3];
    string1 = '';
    string2 = '';
    string3 = '';
    
end

networktypes = {' AVGPOOL', ' MAXPOOL', ' VS RIGHT', ' VS CENTER', ' VS CUSTOM'};
stringset3 = {};

for a=1:1:3
    repmat1 = repmat(networktypes(1,a),12,1);
    stringset3 = [stringset3;repmat1];
     
end

label1 = [basestrings,stringset2,stringset3];
label2 = {};

for n=1:1:60
    labeltemp = strjoin(label1(n,:));
    label2 = [label2; labeltemp];
    
end

%Accuracy
acc = csvread('acc.csv');

%Loss
loss = csvread('loss.csv');

%Val_acc
val_acc = csvread('val_acc.csv');

%Val_loss
val_loss = csvread('val_loss.csv');

%FPR
fpr = csvread('fpr(7).csv');

%TPR
tpr = csvread('tpr(8).csv');

fpr1 = [];
tpr1 = [];
fprs = [];
tprs = [];
fprs2 = [];
tprs2 = [];
fprindex = 2;

for a=2:1:16
    
    for n=1:1:4
            
        truth = 0;
        while truth == 0;    
            if (fpr(fprindex, a) == 1) & (fpr(fprindex+1, a) == 0)
                truth = 1;
                
            end
            
            
            (fpr(fprindex, a) ~= 1) & (fpr(fprindex, a) ~= 0)
            fpr1 = [fpr1, fpr(fprindex, a)];
            tpr1 = [tpr1, tpr(fprindex, a)];
            fprindex = fprindex + 1;

        end

        fpr1 = [fpr1, fpr(fprindex, a)];
        tpr1 = [tpr1, tpr(fprindex, a)];
        fprindex = fprindex + 1;

        fprs = [fprs, fpr1];
        fpr1 = [];
        tprs = [tprs, tpr1];
        tpr1 = [];

    end
    
    fprindex = 2;
    
    fprs2 = [fprs2, fprs];
    tprs2 = [tprs2, tprs];
    
    fprs = [];
    tprs = [];

end

fprbreaks = [1];

for n=1:1:1007
    if (fprs2(1,n) == 1) & (fprs2(1,n+1) == 0);
        fprbreaks = [fprbreaks, n];
        
    end

end

%AUC
auc = csvread('auc(8).csv');
auc1 = [];

for b=2:1:16
    for a=2:1:5
        auc1 = [auc1, auc(a,b)];

    end
end

%Writes

for n=1:1:60
    range = ['A', num2str(11*n-10)];
    range1 = ['A', num2str(11*n-9)];
    range2 = ['A', num2str(11*n-8)];
    range3 = ['A', num2str(11*n-7)];
    range4 = ['A', num2str(11*n-6)];
    range5 = ['A', num2str(11*n-5)];
    range6 = ['A', num2str(11*n-4)];
    range7 = ['A', num2str(11*n-3)];
    range8 = ['A', num2str(11*n-2)];
    writecell(label2(n,1), 'Test.xlsx', 'Range', range);
    writecell({'ACC'}, 'Test.xlsx', 'Range', range1);
    writecell({'LOSS'}, 'Test.xlsx', 'Range', range2);
    writecell({'VAL_ACC'}, 'Test.xlsx', 'Range', range3);
    writecell({'VAL_LOSS'}, 'Test.xlsx', 'Range', range4);
    writecell({'FPR'}, 'Test.xlsx', 'Range', range5);
    writecell({'TPR'}, 'Test.xlsx', 'Range', range6);
    writecell({'AUC'}, 'Test.xlsx', 'Range', range7);
    writecell({'TIME'}, 'Test.xlsx', 'Range', range8);
    
    %Actual Data
    writematrix(acc(n,:), 'Test.xlsx', 'Range', [['B', num2str(11*n-9)], ':', 'EU', range1]);
    writematrix(loss(n,:), 'Test.xlsx', 'Range', [['B', num2str(11*n-8)], ':', 'EU', range2]);
    writematrix(val_acc(n,:), 'Test.xlsx', 'Range', [['B', num2str(11*n-7)], ':', 'EU', range3]);
    writematrix(val_loss(n,:), 'Test.xlsx', 'Range', [['B', num2str(11*n-6)], ':', 'EU', range4]);
    [meme2, size1] = size(tprs2(fprbreaks(n):fprbreaks(n+1)));
    writematrix(fprs2(fprbreaks(n)+1:fprbreaks(n+1)),'Test.xlsx', 'Range', [['B', num2str(11*n-5)], ':', idxexcel(size1+1), range5])
    writematrix(tprs2(fprbreaks(n)+1:fprbreaks(n+1)),'Test.xlsx', 'Range', [['B', num2str(11*n-4)], ':', idxexcel(size1+1), range6])
    writematrix(auc1(1,n), 'Test.xlsx', 'Range', ['B', num2str(11*n-3)]);
    
end

%%% convert index to A1 notation
%%% Written by Matt Brunner
%%% https://www.mathworks.com/matlabcentral/fileexchange/28794-convert-index-to-excel-a1-notation

function a1String = idxexcel(idx)

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

if idx < 27
a1String = alphabet(idx);
else
idx2 = rem(idx,26);
if idx2 == 0
a1String = [alphabet(floor(idx/26)-1),'Z'];
else
a1String = [alphabet(floor(idx/26)),alphabet(idx2)];
end
end
end
%%%
