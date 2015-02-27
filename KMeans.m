function [U, E_in] = KMeans(data, K)
[N, d] = size(data);
% init U
sampleIds = randsample(1:N, K, false);
U = data(sampleIds, :);
labels_u = zeros(N, 1);
while true
    stop = true;
    for i = 1:N
        x = data(i, :);
        % check label
        label = 0;
        dist = 0;
        for j = 1:K
            tmp_dist = sum((x-U(j, :)).^2);
            if label == 0 || tmp_dist < dist
                label = j;
                dist = tmp_dist;
            end
        end
        if labels_u(i) ~= label
            stop = false;
        end
        labels_u(i) = label;
    end
    if stop == true
        break;
    end
    %update U
    new_U = zeros(K, d);
    labels_count = zeros(K, 1);
    for i = 1:N
        label = labels_u(i);
        new_U(label, :) = new_U(label, :) + data(i, :);
        labels_count(label) = labels_count(label) + 1;
    end
    for i = 1:K
        new_U(i, :) = new_U(i, :)/labels_count(i);
    end
    U = new_U;
end
E_in = 0;
for i = 1:N
    label = labels_u(i);
    u = U(label, :);
    E_in = E_in + norm(x-u);
end
E_in = E_in/N;
end