clear;
clc;

%Data Collection and Processing
str = fileread('RESULTSMODDED.txt');
lines = regexp(str, '\r\n|\r|\n', 'split');

loss1 = [];
accuracy = [];
val_loss1 = [];
val_accuracy = [];
meme = [];

%Extract all output lines from code
for n=1:1:21636
    testdata = lines{1,n};
     
    if contains(testdata, '[==============================]') == 1
        meme = [meme, testdata];
        
    end
end

%Extract loss, accuracy, val_loss, val_accuracy
lossindex1 = strfind(meme,'loss:');
accindex1 = strfind(meme,'accuracy:');
val_lossindex1 = strfind(meme,'val_loss:');
val_accindex1 = strfind(meme,'val_accuracy:');

%Remove duplicates
val_accindexdup = val_accindex1 + 4;
val_lossindexdup = val_lossindex1 + 4;
accintersect = intersect(val_accindexdup, accindex1);
lossintersect = intersect(val_lossindexdup, lossindex1);

%CHECK SMALL LOSSES!

accindex = accindex1;
lossindex = [];

val_lossindex = [];
val_accindex = [];

counter4 = 0;

memeval_acc = [];
memeval_loss = []; 

for n = 1:1:9000
    memeval_acc = [memeval_acc,meme(accintersect(1,n)-4:accintersect(1,n)+15)];
    memeval_loss = [memeval_loss,meme(lossintersect(1,n)-4:lossintersect(1,n)+11)];    
       
end

meme_acc = [];
meme_loss = []; 

accindex1 = strfind(meme,' accuracy:');
lossindex1 = strfind(meme,' loss:');

for n = 1:1:9000
    meme_acc = [meme_acc,meme(accindex1(1,n)+1:accindex1(1,n)+16)];
    meme_loss = [meme_loss,meme(lossindex1(1,n)+1:lossindex1(1,n)+12)];    
       
end

%Extract numbers from lists
acc1 = [];
loss1 = [];
val_acc1 = [];
val_loss1 = [];




for n=1:1:9000
    acc1 = [acc1, str2double(meme_acc(16*n-5:16*n))];
    loss1 = [loss1, str2double(meme_loss(12*n-5:12*n))];
    val_acc1 = [val_acc1, str2double(memeval_acc(20*n-5:20*n))];
    val_loss1 = [val_loss1, str2double(memeval_loss(16*n-5:16*n))];
    
end


for n=1:1:9000
    if loss1(n) >= 1
        loss1(n) = loss1(n)*(10^-4);
        
    end
 
end


acc = [];
loss = [];
val_acc = [];
val_loss = [];

for n=1:1:60
    acc = [acc; acc1(1, 150*(n-1)+1:150*n)];
    loss = [loss; loss1(1, 150*(n-1)+1:150*n)];
    val_acc = [val_acc; val_acc1(1, 150*(n-1)+1:150*n)];
    val_loss = [val_loss; val_loss1(1, 150*(n-1)+1:150*n)];
    
end

writematrix(acc,'acc.csv') 
writematrix(loss,'loss.csv') 
writematrix(val_acc,'val_acc.csv') 
writematrix(val_loss,'val_loss.csv') 
