function bt_fit(chain_length, tfs_file, outputfile)
    % fits a single subject's data using braintrak from `tfs_file` and saves the fits to `outputfile`

    % TEMPORARY:
    gp='/external/rprshnas01/netdata_kcni/jglab/MemberSpaces/tm/sleep-modelling';
    cd(gp);
    
    disp(['tfs: ',tfs_file]);
    disp(['allres: ',outputfile]);

    addpath('.');
    addpath(genpath('./code/braintrak'));
    addpath(genpath('./code/corticothalamic-model'));

    % load the data
    d = load(tfs_file);
    d.colheaders = [d.colheaders];

    % set up the model
    model = bt.model.full;
    model.set_electrodes(d);

    % set up the fitting parameters
    use_prior = false; %%%%%%%%%%%%%%%
    debugmode = 0;

    % set up the parallel pool
    sz = str2num(getenv('SLURM_CPUS_PER_TASK'));
    if isempty(sz), sz = 6; end
    p = parpool('local',sz);

    % start the clock and fit
    tic;     
    disp('started fitting at time:'); disp(datetime('now'));

    for npts_per_fit=[chain_length]
        disp('')
        disp(datetime('now'))
        dlen = size(d.s, 1);
        
        %% j is the iterative index of the window being fitted
        for j = 1:dlen
            target_P = squeeze(d.s(j,:))'; % dimensions are: ((n_freqs, n_epochs) ////// , n_channels))
            target_state=cell(1,60);
            if j==1
                [initial_params,initial_pp] = model.initialize_fit(d.f,target_P);
                [~,fit_data(j),plot_data(j)] = bt.core.fit_spectrum(model,d.f(:),target_P,initial_pp,initial_params,npts_per_fit,target_state{j},[],debugmode);
                fdata = bt.feather(model,fit_data(j),plot_data(j),j);
            else                    
                fprintf('Fitting: %d/%d\n',j,dlen)
                target_state{j} = 'NA';

                if use_prior
                    [~,fit_data(j),plot_data(j)] = bt.core.fit_spectrum(model,d.f(:),target_P,fit_data(j-1).posterior_pp,fit_data(j-1).fitted_params,npts_per_fit,target_state{j},[],debugmode);			
                else
                    [~,fit_data(j),plot_data(j)] = bt.core.fit_spectrum(model,d.f(:),target_P,initial_pp,fit_data(j-1).fitted_params,npts_per_fit,target_state{j},[],debugmode);			
                end
                fdata.insert(fit_data(j),plot_data(j),j);
            end
        end

        fdata = fdata.compress();

        tictoc = toc;
        tictoc

        fdata_fd = fdata.fit_data;

        fdata_mod = fdata.model;

        fdata_t = fdata.time;

        save(outputfile, 'fdata', 'fdata_fd','fdata_mod', 'fdata_t', 'tictoc', '-v7.3');

        % save(outfile_2, 'fdata_fd');% '-v7.3');
        
       disp(['finished for outfile at time:']);disp(datetime('now')); disp('')
    end
end
