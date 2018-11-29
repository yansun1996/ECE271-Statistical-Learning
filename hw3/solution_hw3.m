%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_set = load('TrainingSamplesDCT_subsets_8.mat');
alpha = load('Alpha.mat');
alpha = alpha.alpha;

for strategy_idx = 1:2
    if strategy_idx == 1
        strategy = load('Prior_1.mat');
    elseif strategy_idx == 2
        strategy = load('Prior_2.mat');
    end
    
    for dataset_idx = 1:4
        if dataset_idx == 1
            d1_BG = train_set.D1_BG;
            d1_FG = train_set.D1_FG;
        elseif dataset_idx == 2
            d1_BG = train_set.D2_BG;
            d1_FG = train_set.D2_FG;
        elseif dataset_idx == 3
            d1_BG = train_set.D3_BG;
            d1_FG = train_set.D3_FG;
        elseif dataset_idx == 4
            d1_BG = train_set.D4_BG;
            d1_FG = train_set.D4_FG;
        end
        
        bayes_error = [];
        mle_error = [];
        map_error = [];
        n_FG = size(d1_FG,1);
        n_BG = size(d1_BG,1);

        % Loop for different alpha
        for alpha_idx = 1:size(alpha,2)
            cov_0 = zeros(64,64);
            for idx = 1:64
               cov_0(idx,idx) = alpha(alpha_idx)*strategy.W0(idx); 
            end

            % FG
            d1_FG_cov = cov(d1_FG) * (n_FG-1)/n_FG;
            tmp2 = inv(cov_0 + (1/n_FG)*d1_FG_cov);
            mu_1_FG = cov_0 * tmp2 * transpose(mean(d1_FG)) + (1/n_FG) * d1_FG_cov * tmp2 * transpose(strategy.mu0_FG);
            cov_1_FG = cov_0 * tmp2 * (1/n_FG) * d1_FG_cov;
            % predictive distribution (normal distribution)
            mu_pred_FG = mu_1_FG;
            cov_pred_FG = d1_FG_cov + cov_1_FG;

            % BG
            d1_BG_cov = cov(d1_BG) * (n_BG-1)/n_BG;
            tmp3 = inv(cov_0 + (1/n_BG)*d1_BG_cov);
            mu_1_BG = cov_0 * tmp3 * transpose(mean(d1_BG)) + (1/n_BG) * d1_BG_cov * tmp3 * transpose(strategy.mu0_BG);
            cov_1_BG = cov_0 * tmp3 * (1/n_BG) * d1_BG_cov;
            % predictive distribution (normal distribution)
            mu_pred_BG = mu_1_BG;
            cov_pred_BG = d1_BG_cov + cov_1_BG;

            % Prior
            num_FG = size(d1_FG,1);
            num_BG = size(d1_BG,1);
            prior_FG = num_FG / (num_FG + num_BG);
            prior_BG = num_BG / (num_FG + num_BG);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Bayes-BDR

            % BDR
            img = imread('cheetah.bmp');
            img = im2double(img);
            % Add paddle
            img = [zeros(size(img,1),2) img];
            img = [zeros(2, size(img,2)); img];
            img = [img zeros(size(img,1),5)];
            img = [img; zeros(5, size(img,2))];

            %%% DCT
            [m,n] = size(img);
            Blocks = ones(m-7,n-7);
            det_cov_FG = det(cov_pred_FG);
            det_cov_BG = det(cov_pred_BG);
            ave_tmp_FG = transpose(mu_pred_FG);
            ave_tmp_BG = transpose(mu_pred_BG);
            inv_tmp_FG = inv(cov_pred_FG);
            inv_tmp_BG = inv(cov_pred_BG);

            % predict
            const_FG = ave_tmp_FG*inv_tmp_FG*transpose(ave_tmp_FG) + log(det_cov_FG) - 2*log(prior_FG);
            const_BG = ave_tmp_BG*inv_tmp_BG*transpose(ave_tmp_BG) + log(det_cov_BG) - 2*log(prior_BG);
            
            for i=1:m-7
                for j=1:n-7
                    DCT = dct2(img(i:i+7,j:j+7));
                    zigzag_order = zigzag(DCT);
                    feature = zigzag_order;
                    g_cheetah = 0;
                    g_grass = 0;
                    % cheetah
                    g_cheetah = g_cheetah + feature*inv_tmp_FG*transpose(feature);
                    g_cheetah = g_cheetah - 2*feature*inv_tmp_FG*transpose(ave_tmp_FG);
                    g_cheetah = g_cheetah + const_FG;
                    % grass
                    g_grass = g_grass + feature*inv_tmp_BG*transpose(feature);
                    g_grass = g_grass - 2*feature*inv_tmp_BG*transpose(ave_tmp_BG);
                    g_grass = g_grass + const_BG;
                    if g_cheetah >= g_grass
                        Blocks(i,j) = 0;
                    end
                end
            end

            % save prediction
            imwrite(Blocks, ['bayes_prediction_alpha_' int2str(alpha_idx) '_dataset_' int2str(dataset_idx) '_strategy_' int2str(strategy_idx) '.jpg']);
            prediction = mat2gray(Blocks);

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
            error1_64 = (count1/count_cheetah_truth) * prior_FG;
            error2_64 = (count2/count_grass_truth) * prior_BG;
            total_error_64 = error1_64 + error2_64;
            bayes_error = [bayes_error total_error_64];

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % ML-BDR

            % ML prediction
            img = imread('cheetah.bmp');
            img = im2double(img);
            % Add paddle
            img = [zeros(size(img,1),2) img];
            img = [zeros(2, size(img,2)); img];
            img = [img zeros(size(img,1),5)];
            img = [img; zeros(5, size(img,2))];

            %%% DCT
            [m,n] = size(img);
            Blocks = ones(m-7,n-7);
            mean_FG = mean(d1_FG);
            mean_BG = mean(d1_BG);
            ave_tmp_FG = mean_FG;
            ave_tmp_BG = mean_BG;
            inv_covFG = inv(d1_FG_cov);
            inv_covBG = inv(d1_BG_cov);
            DcovFG = det(d1_FG_cov);
            DcovBG = det(d1_BG_cov);

            %%% predict
            const_FG = ave_tmp_FG*inv_covFG*transpose(ave_tmp_FG) + log(DcovFG) - 2*log(prior_FG);
            const_BG = ave_tmp_BG*inv_covBG*transpose(ave_tmp_BG) + log(DcovBG) - 2*log(prior_BG);
            
            for i=1:m-7
                for j=1:n-7
                    DCT = dct2(img(i:i+7,j:j+7));
                    zigzag_order = zigzag(DCT);
                    feature = zigzag_order;
                    g_cheetah = 0;
                    g_grass = 0;
                    % cheetah
                    g_cheetah = g_cheetah + feature*inv_covFG*transpose(feature);
                    g_cheetah = g_cheetah - 2*feature*inv_covFG*transpose(ave_tmp_FG);
                    g_cheetah = g_cheetah + const_FG;
                    % grass
                    g_grass = g_grass + feature*inv_covBG*transpose(feature);
                    g_grass = g_grass - 2*feature*inv_covBG*transpose(ave_tmp_BG);
                    g_grass = g_grass + const_BG;
                    if g_cheetah >= g_grass
                        Blocks(i,j) = 0;
                    end
                end
            end

            %%% save prediction
            imwrite(Blocks, ['mle_prediction_alpha_' int2str(alpha_idx) '_dataset_' int2str(dataset_idx) '_strategy_' int2str(strategy_idx) '.jpg']);
            prediction = mat2gray(Blocks);

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
            error1_64 = (count1/count_cheetah_truth) * prior_FG;
            error2_64 = (count2/count_grass_truth) * prior_BG;
            total_error_64 = error1_64 + error2_64;
            mle_error = [mle_error total_error_64];

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % MAP-BDR

            % BDR
            img = imread('cheetah.bmp');
            img = im2double(img);
            % Add paddle
            img = [zeros(size(img,1),2) img];
            img = [zeros(2, size(img,2)); img];
            img = [img zeros(size(img,1),5)];
            img = [img; zeros(5, size(img,2))];

            %%% DCT
            [m,n] = size(img);
            Blocks = ones(m-7,n-7);
            det_cov_FG = det(d1_FG_cov);
            det_cov_BG = det(d1_BG_cov);
            ave_tmp_FG = transpose(mu_pred_FG);
            ave_tmp_BG = transpose(mu_pred_BG);

            % predict
            const_FG = ave_tmp_FG*inv_covFG*transpose(ave_tmp_FG) + log(det_cov_FG) - 2*log(prior_FG);
            const_BG = ave_tmp_BG*inv_covBG*transpose(ave_tmp_BG) + log(det_cov_BG) - 2*log(prior_BG);
            
            for i=1:m-7
                for j=1:n-7
                    DCT = dct2(img(i:i+7,j:j+7));
                    zigzag_order = zigzag(DCT);
                    feature = zigzag_order;
                    g_cheetah = 0;
                    g_grass = 0;
                    % cheetah
                    g_cheetah = g_cheetah + feature*inv_covFG*transpose(feature);
                    g_cheetah = g_cheetah - 2*feature*inv_covFG*transpose(ave_tmp_FG);
                    g_cheetah = g_cheetah + const_FG;
                    % grass
                    g_grass = g_grass + feature*inv_covBG*transpose(feature);
                    g_grass = g_grass - 2*feature*inv_covBG*transpose(ave_tmp_BG);
                    g_grass = g_grass + const_BG;
                    if g_cheetah >= g_grass
                        Blocks(i,j) = 0;
                    end
                end
            end

            % save prediction
            imwrite(Blocks, ['map_prediction_alpha_' int2str(alpha_idx) '_dataset_' int2str(dataset_idx) '_strategy_' int2str(strategy_idx) '.jpg']);
            prediction = mat2gray(Blocks);

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
            error1_64 = (count1/count_cheetah_truth) * prior_FG;
            error2_64 = (count2/count_grass_truth) * prior_BG;
            total_error_64 = error1_64 + error2_64;
            map_error = [map_error total_error_64];
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % plot
        plot(alpha,bayes_error,alpha,mle_error,alpha,map_error);
        legend('Predict','ML','MAP');
        set(gca, 'XScale', 'log');
        title('PoE vs Alpha');
        xlabel('Alpha');
        ylabel('PoE');
        saveas(gcf,['Strategy_' int2str(strategy_idx) '_dataset_' int2str(dataset_idx) '_PoEvsAlpha.png']);
    end
end
