function [] = drawMat_ioudice(fileName)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
data = load(fileName);
ana_fig_root = './res_fig/';
if exist(ana_fig_root)== 0
    mkdir(ana_fig_root)
end
matname = regexp(fileName, '/', 'split');
matname = matname(end);
pure_name= regexp(matname, '\.', 'split');
pure_name = pure_name(1);
loss_name = regexp(pure_name, '_', 'split');
loss_name = loss_name(end); %根据文件名获取loss名称
data = getfield(data, 'data');
dsize = size(data);
epochn = dsize(1);
endIndex = dsize(2);
resDimen = int16(epochn * endIndex / 1000)
[miou_re, mdice_re, iou0_re, ~, iou1_re, ~] = DNWanaly_ioudice(fileName);
iter_List = transpose(linspace(1,double(resDimen),resDimen));
plot(iter_List, miou_re, "b-", iter_List, mdice_re, "g-.", iter_List, iou0_re, "r-", iter_List, iou1_re, "y-");
xlabel("iteration/1K");
% title(strcat('Results by loss ',loss_name));
legend('miou', 'mdice', 'iou0', 'iou1');
fig_name = sprintf('f%s%s.fig',date,loss_name)
fig_path = sprintf("%s%s",ana_fig_root, fig_name);
saveas(gcf, fig_path);
end