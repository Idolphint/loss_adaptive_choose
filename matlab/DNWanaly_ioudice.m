function[miou_record, mdice_record, iou0_record, dice0_record, iou1_record, dice1_record] = DNWanaly_ioudice(file)
data = load(file);
data = getfield(data, 'data');
dsize = size(data);
epochn = dsize(1);
endIndex = dsize(2);
resDimen = int16((epochn * endIndex) / 1000);
miou_record = zeros(resDimen,1);
mdice_record = zeros(resDimen,1);
iou0_record = zeros(resDimen,1);
dice0_record = zeros(resDimen,1);
iou1_record = zeros(resDimen,1);
dice1_record = zeros(resDimen,1);
cnt=0;
mk=0;
miou=0;mdice=0;iou0=0;dice0=0;iou1=0;dice1=0;
for ei=1:epochn
    for iter = 1:endIndex
        cnt = cnt+1;
        miou = miou+data(ei, iter, 1);
        mdice = mdice+data(ei, iter, 2);
        iou0 = iou0+data(ei, iter, 3);
        dice0 = dice0+data(ei, iter, 4);
        iou1 = iou1+data(ei, iter, 5);
        dice1 = dice1+data(ei, iter, 6);
        
        if(mod(cnt,1000) == 0)
            mk= mk+1;
            if (mk > resDimen)
                outs="ณฌณ๖มหฃก"
                ei
                iter
                cnt
            end
            miou_record(mk) = miou / 1000;
            mdice_record(mk) = mdice / 1000;
            iou0_record(mk) = iou0 / 1000;
            dice0_record(mk) = dice0 / 1000;
            iou1_record(mk) = iou1 / 1000;
            dice1_record(mk) = dice1 / 1000;
            miou=0;mdice=0;iou0=0;dice0=0;iou1=0;dice1=0;
        end
                    
    end
end
end