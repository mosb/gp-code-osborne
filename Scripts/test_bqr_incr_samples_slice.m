clear;
% cd ~/Code/GP/BQR
% 
% load test_bqr_incr_samples_slice
BQR = [];BQ=[];BMC=[];MC=[];

prob_bqr_incr_samples;

        
% samples = slicesample(prior.means, max_num_samples,'pdf', p_r_fn,'width', prior.sds);
% 
%     clf;hold on;
% ezplot(@(x) p_r_fn(x));
% ff=[];qq=[];for i=1:max_num_samples;ff(i)=p_r_fn(samples(i));qq(i)=q_fn(samples(i));end;
% plot(samples,ff,'b.','MarkerSize',5)
% plot(samples,qq,'r.','MarkerSize',5)
% axis tight
% 
% figure
% histfit(samples,50)
% h = get(gca,'Children');
% set(h(2),'FaceColor',[.8 .8 1])





q = [];
r = [];
for trial = 1:max_trials
    fprintf('trial = %u\n', trial);
    
    BQR = [BQR;nan(1, max_num_samples)];
    BQ = [BQ;nan(1, max_num_samples)];
    BMC = [BMC;nan(1, max_num_samples)];
    MC = [MC;nan(1, max_num_samples)];
    
    samples = slicesample(prior.means, max_num_samples,...
        'pdf', p_r_fn,'width', prior.sds);

    q= [];
    r= [];
    gpq = [];
    gpr = [];
    for num_sample = 1:max_num_samples;
        fprintf('%g,',num_sample);
           
        % structs for q and r no longer reqd
        q = [q;q_fn(samples(num_sample,:))];
        r = [r;r_fn(samples(num_sample,:))];
        
        sample_struct.samples = samples(1:num_sample,:);
        sample_struct.log_r = log(r);
        sample_struct.q = q;
        
%         [r_noise_sd, r_input_scales, r_output_scale] = ...
%             hp_heuristics(samples, r, 100);


        gpq = train_gp('sqdexp', 'constant', gpq, ...
            samples(1:num_sample,:), q, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpq);
        
        q_gp.quad_output_scale = best_hypersample_struct.output_scale;
        q_gp.quad_input_scales = best_hypersample_struct.input_scales;
        q_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        
        gpr = train_gp('sqdexp', 'constant', gpr, ...
            samples(1:num_sample,:), r, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpr);
        
        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
   
        [BQR(trial,num_sample), dummy, BQ(trial,num_sample)] = ...
            predict(sample_struct, prior, r_gp, q_gp);
                
        BMC(trial,num_sample) = predict_BMC(sample_struct, prior, r_gp);
        
        MC(trial,num_sample) = predict_MC(sample_struct, prior);
        
        
    end
    save test_bqr_incr_samples_slice

    
    for i = 1:size(MC,1);

        perf_BQR = sqrt(((BQR(i,end) - exact).^2));
        perf_BQ = sqrt((abs(BQ(i,end) - exact).^2));
        perf_BMC = sqrt((abs(BMC(i,end) - exact).^2));
        perf_MC = sqrt((abs(MC(i,end) - exact).^2));
        std_BQR = sqrt(std((BQR(i,:) - exact).^2));
        std_BQ = sqrt(std((BQ(i,:) - exact).^2));
        std_BMC = sqrt(std((BMC(i,:) - exact).^2));
        std_MC = sqrt(std((MC(i,:) - exact).^2));
        fprintf('Dimension %u\n performance\n BQR:\t%g\t+/-%g\n BQ:\t%g\t+/-%g\n BMC:\t%g\t+/-%g\n MC:\t%g\t+/-%g\n',...
            i,perf_BQR,std_BQR,perf_BQ,std_BQ,perf_BMC,std_BMC,perf_MC,std_MC);

        figure;hold on;
        plot(exact+0*MC(i,:),'k')
        plot(BQR(i,:),'r')
        plot(BMC(i,:),'b')
        plot(MC(i,:),'m')

    end
            
        f = all(~isnan(MC'))';
    
    
            figure;hold on;
        plot(exact+0*MC(1,:),'k')
        plot(mean(BQR(f,:)),'r')
        plot(mean(BMC(f,:)),'b')
        plot(mean(MC(f,:)),'m')
    
    
    fprintf('\n');
end

