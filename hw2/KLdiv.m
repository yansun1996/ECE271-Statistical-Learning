function KLD=KLdiv(P,Q)
    P = P/sum(P);
    Q = Q/sum(Q);
    index = P > 0;
    kld = -1 * P(index).* log(Q(index)./P(index));
    kld(isinf(kld)|isnan(kld)) = 0;
    KLD = sum(kld);
end