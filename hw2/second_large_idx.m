function ind2 = second_large_idx(arr)
    [max1, ind1] = max(arr);
    arr(ind1)      = -Inf;
    [max2, ind2] = max(arr);
end