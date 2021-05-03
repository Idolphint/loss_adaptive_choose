function[]=drawiou0_CMP(respath)
ana_fig_root = '../res_fig/';
l2mat='../res/20210411/DNW_L2.mat';
cemat='../res/20210411/DNW_ce.mat';
focalmat='../res/20210411/DNW_focal.mat';
dicemat='../res/20210411/DNW_dice.mat';

matname = regexp(respath, '/', 'split');
matname = matname(end);
pure_name= regexp(matname, '\.', 'split');
pure_name = pure_name(1);
loss_name = regexp(pure_name, '_', 'split');
loss_name = loss_name(end); %根据文件名获取loss名称

[l2_miou,~,~,~,~,~] = DNWanaly_ioudice(l2mat);
[ce_miou,~,~,~,~,~] = DNWanaly_ioudice(cemat);
[focal_miou,~,~,~,~,~] = DNWanaly_ioudice(focalmat);
[dice_miou,~,~,~,~,~] = DNWanaly_ioudice(dicemat);
[my_miou,~,~,~,~,~] = DNWanaly_ioudice(respath);
[len,~]=size(l2_miou);
iter_list=1:1:len;
iter_list = transpose(iter_list);
plot(iter_list, l2_miou, iter_list, ce_miou, iter_list, focal_miou, iter_list, dice_miou, iter_list, my_miou, '--');
legend('L2', 'ce', 'focal', 'dice', 'ours');
xlabel("1000 iter");
ylabel('miou');
fig_name = sprintf('miouCMP%s%s.fig',date, loss_name)
fig_path = sprintf("%s%s",ana_fig_root, fig_name);
saveas(gcf, fig_path);
end
