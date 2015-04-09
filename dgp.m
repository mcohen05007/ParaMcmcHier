function Data = dgp()
N = 1200;                            % Number of HH
J = 3;                              % Number of Payment types
a = 49.5;		
b = 50.5;
T = round(a+(b-a)*rand(1,N));       % Number of transactions per household

thetam = [.5 .25 -1.5]';                % Marginal utility of expenditure for each payment type
k = numel(thetam);
P = rand(sum(T),J);               % Price
sd = 0.1*eye(k);


% Index identifying which houshold made the transactions
idx = [0 cumsum(T)]; 
dx = zeros(size(P,1),1);
for ii = 1:N
	dx((idx(ii)+1):idx(ii+1))=ii;
end

Data = cell(1,N);
thetai = thetam*ones(1,N) + chol(sd)*randn(k,N);
for i = 1:N
    % Compute Probabilities
    p = log(P(dx==i,:))';
    X = [repmat([eye(J-1);zeros(1,J-1)],sum(dx==i),1) p(:)];
    eu = reshape(exp(X*thetai(:,i)),J,T(i));
    Pr = eu./(ones(J,1)*sum(eu,1));
    % Draw Choices
    Y = mnrnd(1,Pr');                    % Expressed as indicator    
    Data{i} = struct('X',X,'y',Y);
end
end