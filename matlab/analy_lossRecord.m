function[loss_idx, loss0_re, loss1_re, loss2_re, loss3_re] = analy_lossRecord(file)

ana_fig_root = '../res_fig/';
data = load(file);
data = getfield(data, 'data');
[iter, ~]=size(data);
loss0_re=zeros(iter,1);
loss1_re=zeros(iter, 1);
loss2_re=zeros(iter,1);
loss3_re=zeros(iter, 1);
loss_idx = zeros(iter,1);
for a=1:iter
    loss_idx(a)=cell2mat(data(a,2));
    loss4_iter = cell2mat(data(a,1));
    loss0_re(a)=loss4_iter(1);
    loss1_re(a)=loss4_iter(2);
    loss2_re(a)=loss4_iter(3);
    loss3_re(a)=loss4_iter(4);
end
marker_0 = find(loss_idx == 0);
marker_1 = find(loss_idx == 1);
marker_2 = find(loss_idx == 2);
marker_3 = find(loss_idx == 3);
iter_list=1:1:iter;
plot(iter_list, loss0_re, 'b-', 'Marker', 's', 'MarkerIndices',marker_0);
hold on;
plot(iter_list, loss1_re, 'g-', 'Marker', 'o', 'MarkerIndices',marker_1);
hold on;
plot(iter_list, loss2_re, 'p-', 'Marker', 's', 'MarkerIndices',marker_2);
hold on;
plot(iter_list, loss3_re, 'y-', 'Marker', 'o', 'MarkerIndices',marker_3);
hold on;
xlabel("1000 iter");
ylabel("loss value");
legend('L2', 'ce', 'focal', 'dice');
fig_name = sprintf('lossRecord%s.fig',date)
fig_path = sprintf("%s%s",ana_fig_root, fig_name);
saveas(gcf, fig_path);
