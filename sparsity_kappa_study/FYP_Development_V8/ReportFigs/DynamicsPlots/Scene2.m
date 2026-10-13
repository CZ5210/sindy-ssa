% Scene2.m — LVLH场景2绘图 (ECI轨迹、LVLH轨迹含目标点、推力加速度)
clear; close all;

scriptDir = fileparts(mfilename('fullpath'));
srcDir  = fullfile(scriptDir, 'Output');
figDir  = fullfile(scriptDir, 'Figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end

%% 图1: ECI 3D轨迹
fprintf('绘制ECI轨迹...\n');
PlotECI(fullfile(srcDir, 'Data_real2.csv'), fullfile(figDir, 'ECI2.pdf'));

%% 图2: LVLH 3D轨迹 + RT平面投影 (含Target)
fprintf('绘制LVLH轨迹...\n');
PlotLVLH2(fullfile(srcDir, 'Data_LVLH2.csv'), [0 -20 0], fullfile(figDir, 'LVLH2.pdf'));

%% 图3: 推力加速度
fprintf('绘制推力加速度...\n');
PlotThrust(fullfile(srcDir, 'Data_real2.csv'), fullfile(figDir, 'Thrust2.pdf'));

fprintf('全部完成。\n');

%% ======================== 辅助函数 ========================

function PlotECI(dataPath, savePath)
    data = readtable(dataPath);
    isMano = strcmp(data.Flag, 'manoeuvre');

    figure('Color', 'w', 'Position', [100 100 800 800]);  hold on;

    % 地球
    [X, Y, Z] = sphere(40);
    surf(X*6371, Y*6371, Z*6371, 'FaceColor', [0.16 0.41 0.74], ...
         'EdgeColor', 'none', 'FaceAlpha', 0.05);

    % XYZ坐标轴
    Axis3D(9000, 1500, 'X,Y,Z');

    % 轨迹
    plot3(data.R_x(~isMano), data.R_y(~isMano), data.R_z(~isMano), ...
          'Color', [0.20 0.60 0.86], 'LineStyle', '--', 'LineWidth', 0.5, ...
          'DisplayName', 'Free Flight');
    plot3(data.R_x(isMano), data.R_y(isMano), data.R_z(isMano), ...
          'Color', [0.80 0.26 0.21], 'LineStyle', '-', 'LineWidth', 0.5, ...
          'DisplayName', 'Manoeuvre');

    h1 = scatter3(data.R_x(1), data.R_y(1), data.R_z(1), 120, ...
             [0.13 0.60 0.33], 'o', 'filled', 'DisplayName', 'Start');
    h2 = scatter3(data.R_x(end), data.R_y(end), data.R_z(end), 120, ...
             [0.83 0.67 0.05], 'd', 'filled', 'DisplayName', 'End');
    uistack([h1 h2], 'top');

    xlabel('X (km)'); ylabel('Y (km)'); zlabel('Z (km)');
    L = 8000;  xlim([-L L]); ylim([-L L]); zlim([-L L]);
    axis equal; grid off;  view(45, 45);
    legend('Location', 'best');

    exportgraphics(gcf, savePath, 'Resolution', 300);
    fprintf('  已保存: %s\n', savePath);
    close;
end

function PlotLVLH2(dataPath, Target, savePath)
    data = readtable(dataPath);
    Rx = data.R_x;  Ry = data.R_y;  Rz = data.R_z;
    isMano = strcmp(data.Flag, 'manoeuvre');

    figure('Color', 'w', 'Position', [100 100 800 800]);
    % === 上: 3D LVLH ===
    subplot(3, 1, 1:2);  hold on;

    Axis3D(5, 0.7, 'R,T,N');
    plot3(Rx, Ry, Rz, 'Color', [0.20 0.60 0.86], 'LineStyle', '--', ...
          'LineWidth', 1.5, 'DisplayName', 'Free Flight');
    plot3(Rx(isMano), Ry(isMano), Rz(isMano), ...
          'Color', [0.80 0.26 0.21], 'LineStyle', '-', ...
          'LineWidth', 2, 'DisplayName', 'Maneuver');

    h1 = scatter3(Rx(1), Ry(1), Rz(1), 120, [0.13 0.60 0.33], 'o', 'filled', ...
             'DisplayName', 'Start');
    h2 = scatter3(Rx(end), Ry(end), Rz(end), 120, [0.83 0.67 0.05], 'd', 'filled', ...
             'DisplayName', 'End');
    h3 = scatter3(Target(1), Target(2), Target(3), 150, [0.5 0 0.5], 'p', 'filled', ...
             'DisplayName', 'Target');
    uistack([h1 h2 h3], 'top');

    xlabel('R (km)','FontSize',12); ylabel('T (km)','FontSize',12); zlabel('N (km)','FontSize',12);
    xlim([-5 5]); ylim([-30 10]); zlim([-5 5]);
    pbaspect([10 40 10]);
    grid on;
    legend('Location', 'northeast','FontSize',14);  view(3);

    % === 下: RT 平面 ===
    subplot(3, 1, 3);  hold on;

    plot(Ry, Rx, 'Color', [0.20 0.60 0.86], 'LineStyle', '--', ...
         'LineWidth', 1, 'DisplayName', 'Free Flight');
    plot(Ry(isMano), Rx(isMano), 'Color', [0.80 0.26 0.21], ...
         'LineStyle', '-', 'LineWidth', 1.5, 'DisplayName', 'Maneuver');

    scatter(Ry(1), Rx(1), 100, [0.13 0.60 0.33], 'o', 'filled', ...
            'DisplayName', 'Start');
    scatter(Ry(end), Rx(end), 100, [0.83 0.67 0.05], 'd', 'filled', ...
            'DisplayName', 'End');
    scatter(Target(2), Target(1), 120, [0.5 0 0.5], 'p', 'filled', ...
            'DisplayName', 'Target');

    Axis2D(3, 0.7, 'R,T');

    xlabel('T (km)','FontSize',12); ylabel('R (km)','FontSize',12);
    xlim([-30 10]); ylim([-5 5]);
    axis equal;  grid on;

    % exportgraphics(gcf, savePath, 'Resolution', 300);
    % fprintf('  已保存: %s\n', savePath);
    % close;
end

function PlotThrust(dataPath, savePath)
    data = readtable(dataPath);
    t = data.Time / 3600;

    figure('Color', 'w', 'Position', [100 100 800 600]);
    acc = {'u_r', 'u_t', 'u_n'};
    for i = 1:3
        subplot(3, 1, i);  hold on;  grid on;
        plot(t, data.(acc{i}), 'k-', 'LineWidth', 2);
        xlabel('Time (h)');
        ylabel([acc{i} '  (km/s^2)']);
        if i == 3, ylim([-1e-8 1e-8]); end
    end
    exportgraphics(gcf, savePath, 'Resolution', 300);
    fprintf('  已保存: %s\n', savePath);
    close;
end

function Axis3D(L, buf, label)
    c = {[0.85 0.33 0.33], [0.33 0.60 0.33], [0.33 0.33 0.85]};
    plot3([0 L], [0 0], [0 0], 'Color', c{1}, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot3([0 0], [0 L], [0 0], 'Color', c{2}, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot3([0 0], [0 0], [0 L], 'Color', c{3}, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    if strcmp(label, 'R,T,N')
        text(L+buf, 0, 0, 'R', 'Color', c{1}, 'FontSize', 12, 'HandleVisibility', 'off');
        text(0, L+buf, 0, 'T', 'Color', c{2}, 'FontSize', 12, 'HandleVisibility', 'off');
        text(0, 0, L+buf, 'N', 'Color', c{3}, 'FontSize', 12, 'HandleVisibility', 'off');
    else
        text(L+buf, 0, 0, 'X', 'Color', c{1}, 'FontSize', 12, 'HandleVisibility', 'off');
        text(0, L+buf, 0, 'Y', 'Color', c{2}, 'FontSize', 12, 'HandleVisibility', 'off');
        text(0, 0, L+buf, 'Z', 'Color', c{3}, 'FontSize', 12, 'HandleVisibility', 'off');
    end
end

function Axis2D(L, buf, label)
    plot([0 L], [0 0], 'Color', [0.33 0.60 0.33], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot([0 0], [0 L], 'Color', [0.85 0.33 0.33], 'LineWidth', 1.5, 'HandleVisibility', 'off');
    if strcmp(label, 'R,T')
        text(0, L+buf, 'R', 'Color', [0.85 0.33 0.33], 'FontSize', 12, 'HandleVisibility', 'off');
        text(L+buf, 0, 'T', 'Color', [0.33 0.60 0.33], 'FontSize', 12, 'HandleVisibility', 'off');
    else
        text(L+buf, 0, 'X', 'Color', [0.33 0.60 0.33], 'FontSize', 12, 'HandleVisibility', 'off');
        text(0, L+buf, 'Y', 'Color', [0.85 0.33 0.33], 'FontSize', 12, 'HandleVisibility', 'off');
    end
end
