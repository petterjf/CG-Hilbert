clear all;
% Solving the system Hx=b using CG, with H being the Hilbert matrix and 
% b = (1,...,1)'.

n = 20;
A = hilb(n);
b = ones(n,1);
x_k = zeros(n,1);
r_k = A*x_k - b;
p_k = -r_k;

for k=1:1000
    if (norm(r_k) < 1e-6)
        break
    end
    alpha_k = r_k'*r_k/(p_k'*A*p_k);
    x_k = x_k + alpha_k*p_k;
    r_k_p = r_k;
    r_k = r_k + alpha_k*A*p_k;
    beta_k = r_k'*r_k/(r_k_p'*r_k_p);
    p_k = -r_k + beta_k*p_k;
end
    
