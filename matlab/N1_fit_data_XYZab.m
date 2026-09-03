%% FieldTrip multi-subject/session EEG -> PSD -> model fit (Warm-Started Version)
% Original paper pipeline. The fit uses the original approximately 1.1-40 Hz
% target grid. External dependencies and the data root are supplied through
% environment variables; see ../README.md.
clear; clc;
warning off;

% --- External dependencies and data paths ---
fieldtripDir = getenv('FIELDTRIP_DIR');
braintrakDir = getenv('BRAINTRAK_DIR');
corticothalamicDir = getenv('CORTICOTHALAMIC_MODEL_DIR');
rootDir = getenv('VANGUARD_ROOT');

if isempty(fieldtripDir) || isempty(braintrakDir) || isempty(corticothalamicDir) || isempty(rootDir)
    error(['Set FIELDTRIP_DIR, BRAINTRAK_DIR, CORTICOTHALAMIC_MODEL_DIR, and ', ...
           'VANGUARD_ROOT before running this script. See README.md.']);
end

addpath(fieldtripDir);
addpath(genpath(braintrakDir));
addpath(genpath(corticothalamicDir));

% --- Dataset root & Output ---
outDir  = fullfile(rootDir, 'Fitted_Parameters Previous Session Start XYZab Alpha emphasized');
if ~exist(outDir, 'dir'), mkdir(outDir); end

% --- Init FieldTrip ---
if exist('ft_defaults','file'), ft_defaults; end

% --- Helpers ---
getNum = @(s) sscanf(regexprep(s,'^\D+',''),'%d');

% --- Subjects, sorted numerically ---
subDirs = dir(fullfile(rootDir,'sub-*'));
subDirs = subDirs([subDirs.isdir]);
[~,idxS] = sort(arrayfun(@(d)getNum(d.name), subDirs));
subDirs   = subDirs(idxS);

% --- Constants ---
condOrder = {'pre','post','post60','post120'};
nRuns     = numel(condOrder);                   % 4
nSess     = 7;                                  % 7 Timepoints total
nSub      = numel(subDirs);
nParams   = 7;

% Sessions 1..5 have 4 runs; Sessions 6..7 have 1 run (mapped to index 1).
exists_mask = false(nRuns, nSess);
exists_mask(:, 1:5) = true;
exists_mask(1, 6:7) = true;

% --- Welch frequency grid ---
freq_ref = (1:0.1:40).';
nF = numel(freq_ref);

totalIters = nSub * nSess * nRuns;
tStart = tic;

fprintf('Root: %s\n', rootDir);

%% PASS 1 & 2: Load -> PSD -> Fit -> Save per subject
for si = 1:nSub
    subID   = subDirs(si).name;
    subPath = fullfile(rootDir, subID, subID);
    if ~exist(subPath, 'dir'), subPath = fullfile(rootDir, subID); end
    
    % --- Prealloc for THIS subject ---
    psd_sub    = nan(nF, nSess, nRuns);
    fits_sub   = nan(nSess, nRuns);
    params_sub = nan(nSess, nRuns, nParams);
    
    sesDirs = dir(fullfile(subPath,'ses-*'));
    sesDirs = sesDirs([sesDirs.isdir]);
    [~,idxSess] = sort(arrayfun(@(d)getNum(d.name), sesDirs));
    sesDirs     = sesDirs(idxSess);
    
    fprintf('\n>> %s (%d sessions found)\n', subID, numel(sesDirs));
    
    for ti = 1:numel(sesDirs)
        sesID   = sesDirs(ti).name;
        eegPath = fullfile(subPath, sesID, 'eeg');
        if ~exist(eegPath,'dir'), continue; end
        
        fifFiles = dir(fullfile(eegPath,'*.fif'));
        if isempty(fifFiles), continue; end
        
        % Build ordered list of files according to condOrder
        orderedFiles = cell(1, nRuns);
        namesLower   = lower(string({fifFiles.name}));
        
        for co = 1:nRuns
            target = condOrder{co};
            hasThis = contains(namesLower, "_task-" + target + "_");
            
            % Special catch for session 6 'post24' -> mapped to 'pre' (co=1)
            if ti == 6 && co == 1 && any(contains(namesLower, "_task-post24_"))
                hasThis = contains(namesLower, "_task-post24_");
            end
            
            % Special catch for session 7 'post7' -> mapped to 'pre' (co=1)
            if ti == 7 && co == 1 && any(contains(namesLower, "_task-post7_"))
                hasThis = contains(namesLower, "_task-post7_");
            end
            
            if any(hasThis)
                idx = find(hasThis, 1, 'first');
                orderedFiles{co} = fullfile(eegPath, fifFiles(idx).name);
            end
        end
        
        for ri = 1:nRuns
            if ~exists_mask(ri, ti) || isempty(orderedFiles{ri}), continue; end
            
            thisFile = orderedFiles{ri};
            
            % --- Load Epoched Data (No Filtering) ---
            cfg = [];
            cfg.dataset = thisFile;
            data = ft_preprocessing(cfg);
            
            % --- Join trials ---
            data_cont = make_continuous_noNaN(data);
            if isempty(data_cont.trial{1}), continue; end
            
            X  = data_cont.trial{1};
            fs = data_cont.fsample;
            
            % --- Welch PSD ---
            win_sec = 5;
            seglen  = max(1, round(win_sec * fs));
            N       = size(X,2);
            
            if N < seglen, continue; end
            
            noverlap = floor(0.5 * seglen);
            [S, f] = pwelch(X.', seglen, noverlap, [], fs);
            if isempty(S) || isempty(f), continue; end
            
            S = max(S, eps);
            goodf = isfinite(f) & all(isfinite(S),2);
            f = f(goodf); S = S(goodf,:);
            if numel(f) < 8, continue; end
            
            psd = mean(S, 2);
            if f(1) == 0, f = f(2:end); psd = psd(2:end); end
            
            if ~(max(psd) > 1e-20 && sum(psd) > 1e-18 && (max(f)-min(f) >= 5)), continue; end
            
            % Interpolate to common grid
            yi = interp1(f, psd, freq_ref, 'linear', 'extrap');
            psd_sub(:, ti, ri) = max(yi, eps);
            
            % =========================================================
            % --- Model Fit (Chronological Warm-Start) ---
            % =========================================================
            yfit = psd_sub(2:end, ti, ri);
            freqs = freq_ref(2:end);
            current_model = bt.model.reduced_alpha_emphasized;
            
            initial_values = []; % Default to empty
            prior_pp = [];
            npts = '60s';
            
            % 1. Determine the immediately preceding chronological timepoint
            warm_ti = [];
            warm_ri = [];
            
            if ri > 1
                % If it's post, post60, or post120 -> use the previous run today
                warm_ti = ti;
                warm_ri = ri - 1;
            elseif ti > 1
                % If it's run 1 (pre) of a new day -> use the last valid run from yesterday
                warm_ti = ti - 1;
                % Search backwards through yesterday's runs to find the last successful one
                for prev_r = nRuns:-1:1
                    if exists_mask(prev_r, warm_ti) && ~isnan(fits_sub(warm_ti, prev_r))
                        warm_ri = prev_r;
                        break;
                    end
                end
            end
            
            % 2. Check if we found a valid chronological predecessor
            if ~isempty(warm_ti) && ~isempty(warm_ri) && ~isnan(fits_sub(warm_ti, warm_ri))
                
                prev_chisq = fits_sub(warm_ti, warm_ri);
                prev_params = squeeze(params_sub(warm_ti, warm_ri, :));
                
                % Strict safety threshold based on your typical good fit (e.g., 0.57)
                chi_sq_threshold = 5.0;
                
                if prev_chisq < chi_sq_threshold
                    % Ensure it's passed as a row vector
                    if size(prev_params, 1) > 1
                        initial_values = prev_params.';
                    else
                        initial_values = prev_params;
                    end
                    fprintf('      [Warm-start: using Ses-%d Run-%d to seed Ses-%d Run-%d]\n', warm_ti, warm_ri, ti, ri);
                end
            end
            
            try
                % Call the core fit_spectrum function directly to pass initial_values
                [a, fit_data, ~] = bt.core.fit_spectrum(current_model, freqs, yfit, prior_pp, initial_values, npts);
                
                fits_sub(ti,ri)      = fit_data.fitted_chisq;
                params_sub(ti,ri,:)  = fit_data.fitted_params;
            catch ME
                warning('Fit failed at (si=%d,ti=%d,ri=%d): %s', si, ti, ri, ME.message);
            end
            % =========================================================
            
        end
    end
    
    % --- Save Subject Data ---
    param_names = a.model.param_names;
    
    saveName = fullfile(outDir, sprintf('%s_model_fits.mat', subID));
    save(saveName, 'psd_sub', 'fits_sub', 'params_sub', 'condOrder', 'freq_ref', 'param_names');
    fprintf('Saved %s\n', saveName);
end

fprintf('\nPipeline Complete.\n');

%% Local helper: join trials w/o inserting zeros
function data_out = make_continuous_noNaN(data_in)
data_out = data_in;
ntr = numel(data_in.trial);
if ntr <= 1, return; end

fs = data_in.fsample;
allX = []; allT = []; t0 = 0;

for k = 1:ntr
    xk = data_in.trial{k};
    tk = data_in.time{k};
    if isempty(xk) || isempty(tk), continue; end
    
    finite_cols = all(isfinite(xk), 1);
    if ~any(finite_cols)
        good_ch = all(isfinite(xk), 2);
        xk = xk(good_ch, :);
        finite_cols = all(isfinite(xk), 1);
    end
    
    xk = xk(:, finite_cols);
    tk = tk(finite_cols);
    if isempty(xk), continue; end
    
    ns = size(xk,2);
    allX = [allX, xk];
    allT = [allT, t0 + (0:ns-1)/fs];
    t0   = t0 + ns/fs;
end

if isempty(allX)
    data_out.trial = {[]}; data_out.time  = {[]};
else
    data_out.trial = {allX}; data_out.time  = {allT};
end
end
