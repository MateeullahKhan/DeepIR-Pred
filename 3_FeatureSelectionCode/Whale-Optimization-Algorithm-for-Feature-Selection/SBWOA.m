function [sFeat,Sf,Nf,curve] = SBWOA(feat,label,N,max_Iter,HO)
% ---------------------------------------------------------
% Spatial Bound Whale Optimization Algorithm (SB-WOA)
% Strict Algorithm-Faithful Implementation
% ---------------------------------------------------------

%% Parameters
lb = 0;
ub = 1;
b  = 1;

fun = @jFitnessFunction;
D   = size(feat,2);   % number of features

%% -------- Spatial Pool (Eq. 5 & 11) ---------------------
SpatialPool = [ ...
    0.05  0.10  0.15  0.20  0.25  0.30  0.35  0.40  0.45  0.50;
    0.025 0.05  0.075 0.10  0.125 0.15  0.175 0.20  0.225 0.25 ];

nSP = size(SpatialPool,2);

%% -------- Spatial Value Initialization (Eq. 12) ---------
SpatialValue = [0 0.111 0.222 0.333 0.444 0.556 0.667 0.778 0.889 1];

%% -------- Whale Initialization (original jWOA style) ----
X = zeros(N,D);
for i = 1:N
    for d = 1:D
        X(i,d) = lb + (ub - lb) * rand();
    end
end

fit  = zeros(1,N);
fitG = inf;
Xgb  = zeros(1,D);

for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > 0.5),HO);
    if fit(i) < fitG
        fitG = fit(i);
        Xgb  = X(i,:);
    end
end

curve = zeros(1,max_Iter);
TS    = zeros(1,N);   % TS(i): spatial set index selected by tournament

t = 1;

%% ================= Main Loop (Algorithm 5) ===============
while t <= max_Iter

    a = 2 - (2 * t / max_Iter);

    %% -------- WOA Position Update (Eq. 1, 5, 8) ----------
    for i = 1:N

        A = 2*a*rand - a;
        C = 2*rand;
        p = rand;
        l = -1 + 2*rand;

        if p < 0.5
            if abs(A) < 1
                % Eq. (1): Encircling prey
                for d = 1:D
                    Dx = abs(C * Xgb(d) - X(i,d));
                    X(i,d) = Xgb(d) - A * Dx;
                end
            else
                % Eq. (8): Random search
                k = randi(N);
                for d = 1:D
                    Dx = abs(C * X(k,d) - X(i,d));
                    X(i,d) = X(k,d) - A * Dx;
                end
            end
        else
            % Eq. (5): Spiral updating
            for d = 1:D
                dist = abs(Xgb(d) - X(i,d));
                X(i,d) = dist * exp(b*l) * cos(2*pi*l) + Xgb(d);
            end
        end

        % Boundary control
        X(i,:) = max(min(X(i,:),ub),lb);
    end

    %% -------- Spatial Bound Control (Algorithms 2 & 3) ---
    for i = 1:N

        % ---- Tournament Selection (k = 3, minimization) ---
        cand = randperm(nSP,3);
        [~,idx] = min(SpatialValue(cand));
        TS(i) = cand(idx);   % index of selected spatial set

        d_max = SpatialPool(1,TS(i));
        d_min = SpatialPool(2,TS(i));

        % Eq. (13) and Eq. (14)
        Dmax = fix(d_max * D);
        Dmin = ceil(d_min * D);

        binX = X(i,:) > 0.5;
        Nx   = sum(binX);

        % -------- Algorithm 2: Negative Selection ----------
        if Nx > Dmax
            idx1 = find(binX == 1);
            rm   = randperm(length(idx1), Nx - Dmax);
            binX(idx1(rm)) = 0;
        end

        % -------- Algorithm 3: Positive Selection ----------
        if Nx < Dmin
            idx0 = find(binX == 0);
            ad   = randperm(length(idx0), Dmin - Nx);
            binX(idx0(ad)) = 1;
        end

        X(i,:) = binX;
    end

    %% -------- Fitness Evaluation -------------------------
    for i = 1:N
        fit(i) = fun(feat,label,(X(i,:) > 0.5),HO);
        if fit(i) < fitG
            fitG = fit(i);
            Xgb  = X(i,:);
        end
    end

    %% -------- Algorithm 4: Spatial Value Update ----------
    % (STRICT nested-loop implementation)
    for i = 1:N
        for j = 1:nSP
            if TS(i) == j
                SpatialValue(j) = (SpatialValue(j) + fit(i)) / 2;
            end
        end
    end

    curve(t) = fitG;
    fprintf('Iteration %d Best (SB-WOA) = %f\n', t, curve(t));

    t = t + 1;
end

%% -------- Final Feature Subset ---------------------------
Pos   = 1:D;
Sf    = Pos(Xgb > 0.5);
Nf    = length(Sf);
sFeat = feat(:,Sf);

end