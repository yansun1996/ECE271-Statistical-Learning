%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem a
prior_cheetah = 250/(1053+250);
prior_grass = 1053/(1053+250);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem b
train_set = load('TrainingSamplesDCT_8.mat');
FGmat = train_set.TrainsampleDCT_FG;
BGmat = train_set.TrainsampleDCT_BG;

%%% foreground
cache1 = [];
prob_fore = zeros(1,64);
for row=1:size(FGmat,1)
    ind2 = second_large_idx(FGmat(row,:));
    cache1 = [cache1, ind2];
end
h1 = histogram(cache1);
h1.Normalization = 'probability';
for i=1:size(h1.Values,2)
    prob_fore(i) = h1.Values(i);
end
saveas(h1, 'foreground_hist', 'jpg');

%%% background
cache2 = [];
prob_back = zeros(1,64);
for row=1:size(BGmat,1)
    ind2 = second_large_idx(BGmat(row,:));
    cache2 = [cache2, ind2];
end
h2 = histogram(cache2);
h2.Normalization = 'probability';
for i=1:size(h2.Values,2)
    prob_back(i) = h2.Values(i);
end
saveas(h2, 'background_hist', 'jpg');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem c
img = imread('cheetah.bmp');
img = im2double(img);
%%% Add paddle
img = [zeros(size(img,1),4) img];
img = [zeros(4, size(img,2)); img];
img = [img zeros(size(img,1),3)];
img = [img; zeros(3, size(img,2))];

%%% DCT
[m,n] = size(img);
Blocks = zeros(m-7,n-7);
for i=1:m-7
    for j=1:n-7
        DCT = abs(dct2(img(i:i+7,j:j+7)));
        zigzag_order = zigzag(DCT);
        index = second_large_idx(zigzag_order);
        Blocks(i,j) = index;
    end
end

%%% Try to find something
yesornot = zeros(1,64);
for idx=1:64
    yes = prob_fore(1, idx) * prior_cheetah;
    no = prob_back(1, idx) * prior_grass;
    if yes >= no
        yesornot(1,idx) = 1;
    end
end
%%%%% Here we found when idx >= 12, the probability to be 
%%%%% cheetah will be larger than the probability to be grass

%%% Predict
results = zeros(m-7, n-7);
p = size(results,1);
q = size(results,2);
for i=1:p
    for j=1:q
        yes = prob_fore(1, Blocks(i,j)) * prior_cheetah;
        no = prob_back(1, Blocks(i,j)) * prior_grass;
        if yes >= no && Blocks(i,j) >= 12
            results(i,j) = 1;
        end
    end
end

%%% save prediction
imwrite(results, 'prediction.jpg');
prediction = mat2gray(results);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% problem d
ground_truth = imread('cheetah_mask.bmp')/255;
x = size(ground_truth, 1);
y = size(ground_truth, 2);
count1 = 0;
count2 = 0;
count_cheetah_truth = 0;
count_grass_truth = 0;
for i=1:x
    for j=1:y
        if prediction(i,j) > ground_truth(i,j)
            count2 = count2 + 1;
            count_grass_truth = count_grass_truth + 1;
        elseif prediction(i,j) < ground_truth(i,j)
            count1 = count1 + 1;
            count_cheetah_truth = count_cheetah_truth + 1;
        elseif ground_truth(i,j) >0
            count_cheetah_truth = count_cheetah_truth + 1;
        else
            count_grass_truth = count_grass_truth + 1;
        end
    end
end
error1 = (count1/count_cheetah_truth) * prior_cheetah;
error2 = (count2/count_grass_truth) * prior_grass;
error = error1 + error2;