function [pa_re, cpa0_re, cpa1_re, mpa_re, miou_re, fwiou_re, dice_re, BWdice_re] = DNWanaly(file)
data = load(file);
data = getfield(data, 'data');
dsize = size(data);
epochn = dsize(1);
endIndex = dsize(2);
%有效信息共8个：pixel_acc, class_pa_0, class_pa_1, mean_pa, mIoU, FWIoU, dice,
%BWdiscore
resDimen = int16(epochn * endIndex / 1000);
pa_re = zeros(resDimen,1);
cpa0_re = zeros(resDimen,1);
cpa1_re = zeros(resDimen,1);
mpa_re = zeros(resDimen,1);
miou_re = zeros(resDimen,1);
fwiou_re = zeros(resDimen,1);
dice_re = zeros(resDimen,1);
BWdice_re = zeros(resDimen,1);
cnt=0;
mk = 0;
pa=0;cpa0=0; cpa1=0;mpa=0;miou=0;fwiou=0;dice=0;BWdice=0;
for ei=1:epochn
    for iter = 1:endIndex
        cnt = cnt+1;
        pa = pa+data(ei, iter, 1);
        cpa0 = cpa0+data(ei, iter, 2);
        cpa1 = cpa1+data(ei, iter, 3);
        mpa = mpa+data(ei, iter, 4);
        miou = miou+data(ei, iter, 5);
        fwiou = fwiou+data(ei, iter, 6);
        dice = dice+data(ei, iter, 7);
        BWdice = BWdice+data(ei, iter, 8);
        if(mod(cnt,1000) == 0)
            mk= mk+1;
            pa_re(mk) = pa / 1000;
            cpa0_re(mk) = cpa0 / 1000;
            cpa1_re(mk) = cpa1 / 1000;
            mpa_re(mk) = mpa / 1000;
            miou_re(mk) = miou / 1000;
            fwiou_re(mk) = fwiou / 1000;
            dice_re(mk) = dice / 1000;
            BWdice_re(mk) = BWdice / 1000;
            pa=0;cpa0=0; cpa1=0;mpa=0;miou=0;fwiou=0;dice=0;BWdice=0;
        end
            
    end
end
end
% iter_List = [1:1000:76000];
% plot(iter_List, miou_ce,"r-",iter_List,miou_dice,"g-",iter_List,miou_focal,"b-",iter_List,miou_exp,"c-",iter_List,miou_L1,"m-",iter_List, miou_L2,"y-");
% hold on
% title("miou change",'FontSize',16)
% ylabel("miou",'FontSize',13)
% xlabel("iteration",'FontSize',13)
% plot(iter_List, dice3_record, "b-", iter_List, dice4_record, "g-.", iter_List, dice5_record, "r:")
% hold on
% xlabel("iteration")
% ylabel("dice_score")
% ylabel("dice score")
% hold off