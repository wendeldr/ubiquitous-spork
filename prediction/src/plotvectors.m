close all;
qresult = train_features .* reshape(proj_queries_weight, 1, 1, []) + reshape(proj_queries_bias,1,1,[]);
kresult = train_features .* reshape(proj_keys_weight, 1, 1, []) + reshape(proj_keys_bias,1,1,[]);
vresult = train_features .* reshape(proj_values_weight, 1, 1, []) + reshape(proj_values_bias,1,1,[]);

function [class0, class1] = split_by_label(data, feature_idx, labels)
    class0 = squeeze(data(labels == 0, feature_idx, :));
    class1 = squeeze(data(labels == 1, feature_idx, :));
end

function plot_feature_vectors(q0, q1, k0, k1, v0, v1, feature_idx, show_legend_colors)
    % Create a new figure
    figure;
    hold on;

    % Get the data range for plane sizing
    all_data = [q0; q1; k0; k1; v0; v1];
    x_range = [min(all_data(:,1)) max(all_data(:,1))];
    y_range = [min(all_data(:,2)) max(all_data(:,2))];
    z_range = [min(all_data(:,3)) max(all_data(:,3))];

    % Create grid for planes
    [X, Y] = meshgrid(x_range, y_range);
    [X2, Z] = meshgrid(x_range, z_range);
    [Y2, Z2] = meshgrid(y_range, z_range);

    % Plot transparent planes
    % XY plane (z=0)
    surf(X, Y, zeros(size(X)), 'FaceColor', 'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % XZ plane (y=0)
    surf(X2, zeros(size(X2)), Z, 'FaceColor', 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    % YZ plane (x=0)
    surf(zeros(size(Y2)), Y2, Z2, 'FaceColor', 'green', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % Define marker types for different features
    markers = {'o', 's', 'd', '^', 'v', '<', '>', 'p', 'h', '*'};
    marker = markers{mod(feature_idx-1, length(markers)) + 1};

    % Plot Query vectors
    scatter3(q0(:,1), q0(:,2), q0(:,3), 5, 'k', marker, 'filled', 'DisplayName', sprintf('Query - Class 0 (Feature %d)', feature_idx));
    scatter3(q1(:,1), q1(:,2), q1(:,3), 5, 'red', marker, 'filled', 'DisplayName', sprintf('Query - Class 1 (Feature %d)', feature_idx));

    % Plot Key vectors
    scatter3(k0(:,1), k0(:,2), k0(:,3), 5, 'b', marker, 'filled', 'DisplayName', sprintf('Key - Class 0 (Feature %d)', feature_idx));
    scatter3(k1(:,1), k1(:,2), k1(:,3), 5, 'g', marker, 'filled', 'DisplayName', sprintf('Key - Class 1 (Feature %d)', feature_idx));

    % Plot Value vectors
    scatter3(v0(:,1), v0(:,2), v0(:,3), 5, 'c', marker, 'filled', 'DisplayName', sprintf('Value - Class 0 (Feature %d)', feature_idx));
    scatter3(v1(:,1), v1(:,2), v1(:,3), 5, 'm', marker, 'filled', 'DisplayName', sprintf('Value - Class 1 (Feature %d)', feature_idx));

    % Add labels and title
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title(sprintf('3D Vector Plot of Feature %d - Q/K/V Comparison', feature_idx));
    grid on;
    view(3); % Set to 3D view
    
    if show_legend_colors
        % Add color legend entries only for the first feature
        legend_entries = {'Query - Class 0 (Black)', 'Query - Class 1 (Red)', ...
                         'Key - Class 0 (Blue)', 'Key - Class 1 (Green)', ...
                         'Value - Class 0 (Cyan)', 'Value - Class 1 (Magenta)'};
        legend(legend_entries, 'Location', 'best');
    else
        legend('Location', 'best');
    end
    
    hold off;
end

% Example usage for multiple features
num_features = 2; % Number of features to plot
for i = 1:num_features
    [q0, q1] = split_by_label(qresult, i, train_labels);
    [k0, k1] = split_by_label(kresult, i, train_labels);
    [v0, v1] = split_by_label(vresult, i, train_labels);
    plot_feature_vectors(q0, q1, k0, k1, v0, v1, i, i == 1);
end

figure;
hold on;
i=1;
a1 = squeeze(qresult(i,1,:));
a2 = squeeze(kresult(i,1,:));

b1 = squeeze(qresult(i,2,:));
b2 = squeeze(kresult(i,2,:));

plot3([0 a1(1)], [0 a1(2)], [0 a1(3)], 'k-', 'LineWidth', 2, 'DisplayName', 'Query - Class 0');
plot3([0 a2(1)], [0 a2(2)], [0 a2(3)], 'r-', 'LineWidth', 2, 'DisplayName', 'Key - Class 1');

plot3([0 b1(1)], [0 b1(2)], [0 b1(3)], 'g-', 'LineWidth', 2, 'DisplayName', 'Query - Class 1');
plot3([0 b2(1)], [0 b2(2)], [0 b2(3)], 'b-', 'LineWidth', 2, 'DisplayName', 'Key - Class 0');

legend('Location', 'best');
hold off;



