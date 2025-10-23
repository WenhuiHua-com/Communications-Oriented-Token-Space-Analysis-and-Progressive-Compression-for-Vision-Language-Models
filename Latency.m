%% Systematic plotting script: map BPP + LDPC (code rate r -> block BLER) to p_pkt and delay metrics
clear; clc; close all;

%% ========== Configuration (modify as needed) ==========
H = 256; Wpix = 256; CH = 1;    % Image resolution and number of channels
C = 10e6;                       % Link rate in bits/s (10 Mbps)
N_ldpc = 1024;                  % LDPC codeword length (bits)

% Methods' BPP (write your provided data as a cell array)
methods = {
    'Ours', [0.05204332139756944, 0.05947808159722222, 0.07434760199652778];
    'BPG',             [0.12597540000000002, 0.3954472, 0.5860521];
    'bmshj',           [0.1959, 0.2616, 0.3670];
    'mbt',             [0.1986, 0.2699, 0.3667];
    'cheng',           [0.1860, 0.2506, 0.3301];
};

% LDPC code rates (as in table) and corresponding block BLER (Table II)
code_rates = [1/2, 2/3, 3/4, 5/6, 1.0];  % r
blk_bler_pts = [0, 0.00159, 0.196, 0.909, 0.95];

% Load sweep: we use relative server load alpha (lambda = alpha * mu_eff)
alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9];

% Deadline thresholds (seconds) used for approximate deadline-miss assessment
deadlines = [0.05, 0.1, 0.5];  % 50 ms, 100 ms, 500 ms

%% ========== Preallocation and computations ==========
num_methods = size(methods,1);
num_rates = length(code_rates);

% Storage structure: method -> each BPP -> each code rate metrics
Results = struct();

for m = 1:num_methods
    name = methods{m,1};
    bpp_list = methods{m,2};
    for ib = 1:length(bpp_list)
        bpp = bpp_list(ib);
        B_info = bpp * H * Wpix * CH;       % Number of information bits
        for ir = 1:num_rates
            r = code_rates(ir);
            p_blk = blk_bler_pts(ir);      % Use BLER directly from table
            K_info = r * N_ldpc;           % Information bits per block
            n_blocks = ceil(B_info / K_info);
            p_pkt = 1 - (1 - p_blk)^n_blocks;   % Packet-level failure probability
            B_tx = B_info / r;             % Total transmitted bits on physical link (including redundancy)
            tx_time = B_tx / C;            % Single transmission time (s)
            if p_pkt >= 1
                E_attempts = Inf;
                E_S = Inf;
            else
                E_attempts = 1/(1 - p_pkt);  % Average number of attempts under unlimited ARQ
                E_S = tx_time * E_attempts;  % Average service time (s)
            end
            mu_eff = 1 / E_S;              % Effective service rate (1/s)
            % Store
            Results.(name)(ib).bpp = bpp;
            Results.(name)(ib).B_info = B_info;
            Results.(name)(ib).rates(ir).r = r;
            Results.(name)(ib).rates(ir).p_blk = p_blk;
            Results.(name)(ib).rates(ir).n_blocks = n_blocks;
            Results.(name)(ib).rates(ir).p_pkt = p_pkt;
            Results.(name)(ib).rates(ir).B_tx = B_tx;
            Results.(name)(ib).rates(ir).tx_time = tx_time;
            Results.(name)(ib).rates(ir).E_S = E_S;
            Results.(name)(ib).rates(ir).mu_eff = mu_eff;
        end
    end
end

% To be executed after generating Results

% Common style
fontsz = 16;
linew = 1.6;
mkSize = 7;
colors = lines(num_methods);

%% Figure 1: p_pkt vs LDPC code rate (one curve per method)
lineStyles = {'-','--',':','-.'};
markers    = {'o','s','d','^','v','>','<','p','h'};

fig1 = figure('Color','w','Position',[150 120 800 520]);
hold on; box on;
for m = 1:num_methods
    name = methods{m,1};
    bpp_list = methods{m,2};
    num_bpp = length(bpp_list);
    for ib_sel = 1:num_bpp
        % Read p_pkt for each ib_sel (for all rates)
        ppkts = arrayfun(@(ir) Results.(name)(ib_sel).rates(ir).p_pkt, 1:num_rates);
        % Choose line style and marker (cycled)
        ls = lineStyles{ mod(ib_sel-1, numel(lineStyles)) + 1 };
        mk = markers{  mod(ib_sel-1, numel(markers)) + 1 };
        % DisplayName includes BPP for identification
        displayName = sprintf('%s BPP=%.3f', name, bpp_list(ib_sel));
        plot(code_rates(1:4), ppkts(1:4), ...
             'LineStyle', ls, 'Marker', mk, 'LineWidth', linew, 'MarkerSize', mkSize, ...
             'DisplayName', displayName, 'Color', colors(m,:));
    end
end

set(gca,'FontSize',fontsz);
xlabel('$\mathrm{LDPC\ code\ rate}\; r$', 'Interpreter','latex', 'FontSize',fontsz);
ylabel('$p_{\mathrm{pkt}}$', 'Interpreter','latex', 'FontSize',fontsz);
% title('Packet-level error vs LDPC code rate (mid BPP)', 'Interpreter','latex', 'FontSize',fontsz);

% xticks only up to 5/6 (first 4 elements)
xticks(code_rates(1:4));
xticklabels({'1/2','2/3','3/4','5/6'});   % Can replace with preferred format
grid on;
legend('Location','northeastoutside', 'FontSize', round(fontsz*0.85));

% Save high-resolution figure
set(fig1,'PaperPositionMode','auto');
print(fig1,'fig1_p_pkt_vs_r_allBPP','-dpng','-r300');


%% Figure 2: mean delay (M/M/1) vs relative load alpha (plot for r=2/3 and r=3/4)
plot_rates_idx = find(ismember(code_rates, [2/3, 3/4]));
for pr = plot_rates_idx
    r = code_rates(pr);
    fig = figure('Color','w','Position',[150 120 800 520]);
    hold on; box on;
    for m = 1:num_methods
        name = methods{m,1};
        bpp_list = methods{m,2};
        % *** Change: iterate all BPP to keep style consistent with Fig.1 ***
        num_bpp = length(bpp_list); % retrieve BPP count
        for ib_sel = 1:num_bpp 
            
            % --- Compute Mean Delay (M/M/1) ---
            if isinf(Results.(name)(ib_sel).rates(pr).E_S)
                mean_delays = nan(size(alpha_list));
            else
                E_S = Results.(name)(ib_sel).rates(pr).E_S;
                mu_eff = 1 / E_S;
                lambdas = alpha_list * mu_eff;
                mean_delays = zeros(size(lambdas));
                for ia = 1:length(lambdas)
                    lambda = lambdas(ia);
                    if lambda >= mu_eff
                        mean_delays(ia) = Inf;
                    else
                        % M/M/1 mean waiting time W = 1 / (mu - lambda)
                        mean_delays(ia) = 1 / (mu_eff - lambda);
                    end
                end
            end
            
            % --- Plot: use same style logic as Fig.1 ---
            % Choose line style and marker (cycled, based on BPP index ib_sel)
            ls = lineStyles{ mod(ib_sel-1, numel(lineStyles)) + 1 };
            mk = markers{  mod(ib_sel-1, numel(markers)) + 1 };
            
            % DisplayName includes BPP for identification (consistent with Fig.1)
            displayName = sprintf('%s BPP=%.3f', name, bpp_list(ib_sel)); 

            % *** Change: use ls and mk instead of [style 'o'], and add Color ***
            plot(alpha_list, mean_delays, ...
                 'LineStyle', ls, 'Marker', mk, 'LineWidth',linew, 'MarkerSize',mkSize, ...
                 'DisplayName', displayName, 'Color', colors(m,:)); 
        end
    end
    
    % --- Chart styling (unchanged) ---
    set(gca,'FontSize',fontsz);
    xlabel('$\alpha$ (load fraction of server capacity)', 'Interpreter','latex', 'FontSize',fontsz); % Recommend latex for greek
    ylabel('Mean delay (s)', 'FontSize',fontsz);
%     title(sprintf('M/M/1 mean delay vs $\\alpha$ ($r=%.3f$)', r), 'Interpreter','latex', 'FontSize',fontsz); % Recommend latex for greek
    set(gca,'YScale','log');
    grid on;
    % *** Change: unify legend font size ***
    legend('Location','northeastoutside', 'FontSize', round(fontsz*0.85)); 
    
    % Save
    fname = sprintf('fig2_mean_delay_r_%g_allBPP', round(r*100)); % add allBPP for clarity
    set(fig,'PaperPositionMode','auto');
    print(fig,fname,'-dpng','-r300');
end


%% Figure 3: T_tx vs E[S] comparison (example for r=2/3) — compare each method and its BPP points
% Predefined line styles and markers (consistent with Fig.1 and Fig.2)
lineStyles = {'-','--',':','-.'};
markers    = {'o','s','d','^','v','>','<','p','h'};

% Redefine colors to improve distinguishability.
num_methods = size(methods,1);
temp_colors = lines(num_methods); 

pr = find(code_rates==2/3); % assume code_rates contains 2/3
fig3 = figure('Color','w','Position',[150 120 800 520]); 
hold on; box on;

% Assume fontsz, linew, mkSize are defined

% Legend handles and labels: initialize empty
method_legend_handles = [];
method_legend_labels = {}; 

% Generic handles for T_tx and E[S]
handle_tx_multimarkers = [];
handle_es = [];

% Representative markers for the legend
legend_markers = markers(1:3); % only take first three markers as representatives

for m = 1:num_methods
    name = methods{m,1};
    bpp_list = methods{m,2}; 
    base_color = temp_colors(m,:); 
    
    % --- 1. Create method legend handles (stacked markers) ---
    for i = 1:length(legend_markers)
        m_i = legend_markers{i};
        % Create fake plot handle using method color, hollow marker
        h_temp = plot(NaN, NaN, m_i, 'MarkerSize', mkSize, 'LineStyle', 'none', ...
                     'MarkerFaceColor', 'none', 'MarkerEdgeColor', base_color, 'LineWidth', linew);
        
        method_legend_handles = [method_legend_handles, h_temp];
        
        % Only the first marker has the method label; others are empty labels
        if i == 1
            method_legend_labels = [method_legend_labels, {name}]; % ensure cell
        else
            method_legend_labels = [method_legend_labels, {''}];
        end
    end
    
    % --- 2. Plot actual data points ---
    for ib = 1:length(bpp_list)
        bpp_val = bpp_list(ib);
        
        % T_tx marker (cycled to distinguish BPP)
        mk = markers{  mod(ib-1, numel(markers)) + 1 };
        
        % Assume Results struct is defined
        tx_time = Results.(name)(ib).rates(pr).tx_time;
        E_S = Results.(name)(ib).rates(pr).E_S;
        if isempty(tx_time) || isnan(tx_time), tx_time = NaN; end
        if isempty(E_S) || isnan(E_S) || isinf(E_S), E_S = NaN; end

        % Plot T_tx (Transfer Time)
        plot(bpp_val, tx_time, mk, 'MarkerSize', mkSize, 'LineStyle', 'none', ...
             'MarkerFaceColor', 'none', 'MarkerEdgeColor', base_color, ...
             'LineWidth', linew);
        
        % Plot E[S] (Service Time)
        plot(bpp_val, E_S, 'x', 'MarkerSize', mkSize+2, 'LineStyle', 'none', ...
             'LineWidth', 1.5, 'Color', base_color);

    end
end

% --- 3. Create generic legend handles for T_tx and E[S] (black markers) ---

% T_tx generic legend: vertically display multiple markers
Ttx_legend_label = '$T$ (Transfer Time)';
legend_labels_tx = {}; 
for i = 1:length(legend_markers)
    m_i = legend_markers{i};
    h_temp = plot(NaN, NaN, m_i, 'MarkerSize', mkSize, 'LineStyle', 'none', ...
                 'MarkerFaceColor', 'none', 'MarkerEdgeColor', 'k', 'LineWidth', linew);
    handle_tx_multimarkers = [handle_tx_multimarkers, h_temp];
    
    % Only the first marker has a label
    if i == 1
        legend_labels_tx = [legend_labels_tx, {Ttx_legend_label}];
    else
        legend_labels_tx = [legend_labels_tx, {''}]; 
    end
end

% E[S] generic legend
handle_es = plot(NaN, NaN, 'x', 'MarkerSize', mkSize+2, 'LineStyle', 'none', ...
                'Color', 'k', 'LineWidth', 1.5);
legend_labels_es = {'$E[S]$ (Service Time)'};


% --- Chart settings ---
set(gca,'XScale','linear'); 
set(gca,'YScale','log');

xlabel('BPP', 'FontSize',fontsz); 
ylabel('Time (s, log scale)', 'FontSize',fontsz);
% title('$T$ vs $E[S]$ as a function of BPP ($r=2/3$)', 'Interpreter','latex', 'FontSize',fontsz);

grid on;

% --- 4. Reorganize legend (generic legend after method legend) ---

final_legend_handles = [method_legend_handles, handle_tx_multimarkers, handle_es];
final_legend_labels = [method_legend_labels, legend_labels_tx, legend_labels_es];


lg = legend(final_legend_handles, final_legend_labels, 'Location','northeastoutside', ...
    'Interpreter','latex', 'NumColumns', 1);
lg.Box = 'on';
lg.FontSize = round(fontsz*0.85);

set(fig3,'PaperPositionMode','auto');
print(fig3,'fig3_Ttx_vs_ES_vs_BPP_final_v6_Stable','-dpng','-r300');

% Completion message
fprintf('Plotting complete: 4 separate images have been saved as PNG (300 dpi).\n');

%% ========== Example results printout ==========
fprintf('\nSample output (method, BPP, r=2/3):\n');
for m = 1:num_methods
    name = methods{m,1};
    bpp_list = methods{m,2};
    for ib = 1:length(bpp_list)
        pr = find(code_rates==2/3);
        rinfo = Results.(name)(ib).rates(pr);
        fprintf('%s | BPP=%.4f | r=%.2f | n=%d | p_blk=%.4f | p_pkt=%.4f | tx_time=%.6fs | E_S=%.6fs\n', ...
            name, Results.(name)(ib).bpp, rinfo.r, rinfo.n_blocks, rinfo.p_blk, rinfo.p_pkt, rinfo.tx_time, rinfo.E_S);
    end
end
