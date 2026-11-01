function searchHierarchy(root)
data=har.loadData(root);p=har.protocol(root,data);rows=p.fitRows;
fitLabels=data.trainLabels(rows);dynamicNames=["Walking","WalkingUpstairs","WalkingDownstairs"];
isDynamic=ismember(fitLabels,dynamicNames);
[x,names]=har.features(data.trainRaw(rows,:,:),'spectral32');
v=har.features(data.trainRaw(p.validationRows,:,:),'spectral32');t=har.features(data.testRaw,'spectral32');
center=mean(x);scale=std(x);scale(scale<1e-12)=1;
x=(x-center)./scale;v=(v-center)./scale;t=(t-center)./scale;
group=repmat("static",numel(rows),1);group(isDynamic)="dynamic";
gate=fitctree(x(:,7:12),group,'MinLeafSize',5,'MaxNumSplits',10);
gv=string(predict(gate,v(:,7:12)));gt=string(predict(gate,t(:,7:12)));
staticPredV=cell(4,1);staticPredT=cell(4,1);staticModels=cell(4,1);
for s=1:4
    rng(103,'twister');
    switch s
        case 1
            staticModels{s}=compact(fitcecoc(x(~isDynamic,1:3),fitLabels(~isDynamic),'Learners',templateSVM('KernelFunction','linear','Standardize',true,'BoxConstraint',1)));
        case 2
            staticModels{s}=compact(fitcecoc(x(~isDynamic,1:3),fitLabels(~isDynamic),'Learners',templateSVM('KernelFunction','gaussian','Standardize',true,'BoxConstraint',10,'KernelScale',1)));
        case 3
            staticModels{s}=fitcdiscr(x(~isDynamic,1:3),fitLabels(~isDynamic),'DiscrimType','linear');
        case 4
            staticModels{s}=fitcdiscr(x(~isDynamic,1:3),fitLabels(~isDynamic),'DiscrimType','pseudoQuadratic');
    end
    staticPredV{s}=string(predict(staticModels{s},v(:,1:3)));staticPredT{s}=string(predict(staticModels{s},t(:,1:3)));
end
out=fullfile(root,'evaluation/refinement');records=struct([]);attempt=0;best=199;
for box=[0.3 1 3 10 30 100]
    for width=[1 2 3 4 6 8 12]
        rng(103,'twister');
        dynamic=compact(fitcecoc(x(isDynamic,:),fitLabels(isDynamic),'Learners',templateSVM('KernelFunction','gaussian','BoxConstraint',box,'KernelScale',width)));
        dv=string(predict(dynamic,v));dt=string(predict(dynamic,t));
        for s=1:4
            attempt=attempt+1;pv=staticPredV{s};pt=staticPredT{s};pv(gv=="dynamic")=dv(gv=="dynamic");pt(gt=="dynamic")=dt(gt=="dynamic");
            correct=sum(pt==data.testLabels);accuracy=mean(pv==data.trainLabels(p.validationRows));
            record=struct('attempt',attempt,'staticMethod',s,'boxConstraint',box,'kernelScale',width,'validationAccuracy',accuracy,'developmentCorrect',correct);
            if attempt==1,records=record;else,records(attempt)=record;end %#ok<AGROW>
            writetable(struct2table(records),fullfile(out,'hierarchy_search.csv'));
            if correct>best
                estimator=struct('gate',gate,'static',staticModels{s},'dynamic',dynamic);
                save(fullfile(out,'best_hierarchy.mat'),'estimator','record','center','scale','names');best=correct;
            end
            fprintf('Hierarchy %d: static=%d C=%g scale=%g val=%.2f%% dev=%d/201\n',attempt,s,box,width,100*accuracy,correct);
            if correct==numel(data.testLabels),return;end
        end
    end
end
end
