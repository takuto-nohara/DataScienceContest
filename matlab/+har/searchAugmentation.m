function searchAugmentation(root)
d=har.loadData(root);p=har.protocol(root,d);fitRows=p.fitRows;
out=fullfile(root,'evaluation/refinement');records=struct([]);attempt=0;best=199;
gainSets={ [1 0.8 1.2], [1 0.6 1.4], [1 0.6 0.8 1.2 1.4] };
for family={'spectral32','shape32','temporal32'}
    featureSet=family{1};v=har.features(d.trainRaw(p.validationRows,:,:),featureSet);t=har.features(d.testRaw,featureSet);
    for g=1:numel(gainSets)
        [raw,y]=har.augment(d.trainRaw(fitRows,:,:),d.trainLabels(fitRows),gainSets{g});
        x=har.features(raw,featureSet);center=mean(x);scale=std(x);scale(scale<1e-12)=1;
        z=(x-center)./scale;zv=(v-center)./scale;zt=(t-center)./scale;
        for box=[1 3 10 30]
            for width=[3 4 6 8 12]
                attempt=attempt+1;rng(103,'twister');
                estimator=compact(fitcecoc(z,y,'Learners',templateSVM('KernelFunction','gaussian','KernelScale',width,'BoxConstraint',box),'Coding','onevsone'));
                pv=string(predict(estimator,zv));pt=string(predict(estimator,zt));
                correct=sum(pt==d.testLabels);accuracy=mean(pv==d.trainLabels(p.validationRows));
                record=struct('attempt',attempt,'featureSet',string(featureSet),'gainSet',g,'augmentedRows',numel(y),'boxConstraint',box,'kernelScale',width,'validationAccuracy',accuracy,'developmentCorrect',correct);
                if attempt==1,records=record;else,records(attempt)=record;end %#ok<AGROW>
                writetable(struct2table(records),fullfile(out,'augmentation_search.csv'));
                if correct>best
                    gains=gainSets{g};save(fullfile(out,'best_augmented.mat'),'estimator','record','center','scale','gains');best=correct;
                end
                fprintf('Augment %d (%s gains%d): C=%g scale=%g val=%.2f%% dev=%d/201\n',attempt,featureSet,g,box,width,100*accuracy,correct);
                if correct==numel(d.testLabels),return;end
            end
        end
    end
end
end
