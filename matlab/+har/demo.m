function app = demo(root, row)
% Interactive MATLAB presentation of the saved proto_v10 pipeline.
if nargin<1,root=fileparts(fileparts(fileparts(mfilename('fullpath'))));end
if nargin<2,row=155;end
d=har.loadData(root);b=har.inspectModel('proto_v10',false);p=har.protocol(root,d);
assert(isequal(d.fingerprint,b.protocol.fingerprint),'Dataset does not match the saved model.');
development=har.predict(b,d.testRaw);validation=har.predict(b,d.trainRaw(p.validationRows,:,:));
devMetric=har.metrics(d.testLabels,development,d.classes);
valMetric=har.metrics(d.trainLabels(p.validationRows),validation,d.classes);
fig=uifigure('Name','行動認識の処理デモ | proto_v10','Position',[80 70 1420 900],'Color',[0.97 0.98 1]);
main=uigridlayout(fig,[4 1]);main.RowHeight={44,38,'1x',36};main.Padding=[16 10 16 10];
head=uilabel(main,'Text','6軸の波形から、行動名が決まるまで','FontSize',24,'FontWeight','bold'); %#ok<NASGU>
toolbar=uigridlayout(main,[1 6]);toolbar.ColumnWidth={85,320,180,180,180,'1x'};toolbar.Padding=0;
uilabel(toolbar,'Text','説明する窓','FontSize',14);
items=compose('%03d | %s',(1:numel(d.testLabels))',d.testLabels);
picker=uidropdown(toolbar,'Items',cellstr(items),'ItemsData',1:numel(items),'Value',row,'ValueChangedFcn',@(~,~)refresh());
uibutton(toolbar,'Text','特徴量のコードを開く','ButtonPushedFcn',@(~,~)edit(fullfile(root,'matlab','+har','features.m')));
uibutton(toolbar,'Text','判定のコードを開く','ButtonPushedFcn',@(~,~)edit(fullfile(root,'matlab','+har','predict.m')));
uibutton(toolbar,'Text','全画面をPNG保存','ButtonPushedFcn',@(~,~)exportPages(fullfile(root,'artifacts','demo')));
uibutton(toolbar,'Text','学習処理のコードを開く','ButtonPushedFcn',@(~,~)edit(fullfile(root,'matlab','+har','train.m')));
tabs=uitabgroup(main);titles={'1 処理の全体像','2 センサ波形','3 特徴量と標準化','4 SVMと多数決','5 決定木の分岐','6 評価'};
pages=gobjects(1,6);for k=1:6,pages(k)=uitab(tabs,'Title',titles{k});end
status=uilabel(main,'FontSize',13,'FontColor',[0.18 0.24 0.35]);
refresh();
app=struct('Figure',fig,'Tabs',tabs,'Pages',pages,'RowPicker',picker,'SelectRow',@selectRow, ...
    'Export',@exportPages,'GetExplanation',@getExplanation);

    function e=getExplanation()
        e=har.explainWindow(b,d.testRaw(picker.Value,:,:));
    end
    function selectRow(i)
        validateattributes(i,{'numeric'},{'scalar','integer','>=',1,'<=',numel(items)});picker.Value=i;refresh();
    end
    function refresh()
        e=getExplanation();i=picker.Value;
        status.Text=sprintf('窓 %d | 正解ラベル: %s | 主SVM: %s | 最終判定: %s | ラベルは表示・評価にだけ使用',i,d.testLabels(i),e.primary,e.final);
        for q=1:6,delete(pages(q).Children);end
        overview(e);signals(e);featurePage(e);decision(e);treePage(e);evaluation();
        drawnow;
    end
    function overview(e)
        g=pageGrid(pages(1),{78,'1x',95});
        note(g,'保存済みのモデルを実行します。入力は1件の6信号×128点。学習し直す処理はありません。');
        ax=uiaxes(g);ax.XLim=[0 10];ax.YLim=[0 10];axis(ax,'off');hold(ax,'on');
        boxes={[0.5 7.2 3.6 1.7],[5.5 7.2 4 1.7],[5.5 4.3 4 1.7],[0.5 4.3 3.6 1.7],[0.5 1.4 3.6 1.7],[5.5 1.4 4 1.7]};
        labels={sprintf('入力波形\n6軸 × 128点 / 50 Hz'),sprintf('特徴量を計算\n平均・揺れ・相関・周期 → 32個'), ...
            sprintf('学習時の尺度で標準化\nz = (x - center) / scale'),sprintf('主RBF SVM\n15組の二者比較 → %s',e.primary), ...
            sprintf('静止姿勢の候補だけ多数決\n主SVM + 決定木 + 線形SVM'),sprintf('最終判定\n%s',e.final)};
        for q=1:6
            v=boxes{q};rectangle(ax,'Position',v,'Curvature',0.08,'FaceColor',[0.88 0.93 1],'EdgeColor',[0.22 0.39 0.65],'LineWidth',1.5);
            text(ax,v(1)+v(3)/2,v(2)+v(4)/2,labels{q},'HorizontalAlignment','center','FontSize',15,'Interpreter','none');
        end
        quiver(ax,[4.1 7.5 5.5 2.3 4.1],[8 7.2 5.1 4.3 2.2],[1.4 0 -1.4 0 1.4],[0 -1.2 0 -1.2 0],0,'Color',[0.2 0.3 0.5],'LineWidth',2,'MaxHeadSize',0.5);
        note(g,'対応する実装: loadData.m → features.m → predict.m → staticVote.m。動作の候補なら主SVMの判定をそのまま採用します。');
    end
    function signals(e)
        g=pageGrid(pages(2),{60,'1x','1x',55});
        note(g,'横軸は時刻。端末のX・Y・Z軸をそのまま表示しています。加速度には重力が含まれ、方角の推定は行いません。');
        time=(0:127)/50;
        for q=1:2
            ax=uiaxes(g);cols=(q-1)*3+(1:3);plot(ax,time,e.raw(:,cols),'LineWidth',1.3);grid(ax,'on');
            xlabel(ax,'時間 [s]');ylabel(ax,'配布データの値');legend(ax,{'X','Y','Z'},'Location','eastoutside');
            if q==1,title(ax,'加速度3軸: 端末の傾きと揺れ');else,title(ax,'角速度3軸: 回転の向きと速さ');end
        end
        note(g,'128/50 = 2.56秒相当の窓です。先頭を0秒とすると最後の点は2.54秒。縦軸は元ファイルの数値を換算せず表示しています。');
    end
    function featurePage(e)
        g=pageGrid(pages(3),{55,'1x',240,45});
        note(g,'32個の数値が分類器への入力です。表のrawValueから学習時のcenterを引き、scaleで割ったzをSVMに渡します。');
        tableUI(g,e.featureTable,{'列','特徴量','rawValue','学習時center','学習時scale','z'}, {45,290,120,120,120,120});
        bottom=uigridlayout(g,[1 2]);bottom.ColumnWidth={'1x','1x'};
        ax=uiaxes(bottom);bar(ax,e.standardized,'FaceColor',[0.2 0.45 0.75]);grid(ax,'on');title(ax,'32特徴量の標準化後の値');xlabel(ax,'特徴量の列番号');ylabel(ax,'z');
        ax=uiaxes(bottom);bar(ax,e.frequency,e.power(:,1));xlim(ax,[0 6]);grid(ax,'on');xline(ax,0.5,'--');xline(ax,3,'--');
        title(ax,sprintf('加速度XのFFT | 帯域パワー割合 = %.5f',e.bandFraction(1)));xlabel(ax,'周波数 [Hz]');ylabel(ax,'|FFT|^2');
        note(g,'列27〜32: 0 Hzを除くパワー合計に対する0.5〜3 Hz帯の割合。値が大きいことだけで歩行と決める処理ではありません。');
    end
    function decision(e)
        g=pageGrid(pages(4),{58,240,'1x',50});
        note(g,sprintf('3つの判定: 主SVM = %s / 決定木 = %s / 線形SVM = %s | 最終 = %s',e.votes(1),e.votes(2),e.votes(3),e.final));
        top=uigridlayout(g,[1 2]);top.ColumnWidth={'1x','1x'};
        ax=uiaxes(top);barh(ax,e.negativeLoss,'FaceColor',[0.2 0.45 0.75]);ax.YTick=1:6;ax.YTickLabel=e.classes;grid(ax,'on');
        title(ax,'主SVM: 最大のNegLossを選択');xlabel(ax,'NegLoss（確率ではない）');
        if e.voteApplied,gate='多数決を適用: 3票中2票以上。3票が別なら主SVMを維持。';else,gate='主SVMが動作を予測: 補助2票は最終判定に使いません。';end
        note(top,sprintf('%s\n\n6行動の組み合わせは6×5÷2 = 15組です。\n下表は保存SVMの実スコア。正のクラスへのスコアとサポートベクトル数を表示します。\n\n特徴量ごとの寄与率を表す表ではありません。',gate));
        tableUI(g,e.pairTable,{'比較する2行動','正のクラス','負のクラス','正のクラスのスコア','二者判定','SV数'},{300,180,180,140,180,70});
        note(g,'RBF SVMはサポートベクトルとの類似度を組み合わせます。15スコアをECOCの損失で集計し、単純な15票の最多得票では決めません。');
    end
    function treePage(e)
        g=pageGrid(pages(5),{80,'1x',75});
        note(g,sprintf('補助の決定木が実際に通った分岐です。主SVMの説明とは区別します。到達した葉: %d / 決定木の予測: %s',e.treeLeaf,e.treeLabel));
        tableUI(g,e.treeTable,{'節点','特徴量','この窓の値','元の尺度のしきい値','進んだ枝'},{80,330,180,200,180});
        note(g,'分岐は標準化後の値で計算し、表では読みやすいように元の尺度へ戻しています。rawValue < rawCutなら左、それ以外は右へ進みます。');
    end
    function evaluation()
        g=pageGrid(pages(6),{82,'1x',110});
        note(g,sprintf('保存モデルを再評価: 開発データ %d/%d = %.2f%% | 検証 %d/%d = %.2f%%', ...
            devMetric.correct,devMetric.total,100*devMetric.accuracy,valMetric.correct,valMetric.total,100*valMetric.accuracy));
        two=uigridlayout(g,[1 2]);two.ColumnWidth={'1x','1x'};
        confusionPlot(two,devMetric,'開発データ: 改良時にラベルを参照済み');confusionPlot(two,valMetric,'検証: 固定した735件');
        note(g,'201/201は開発データに対する結果です。改良にラベルを使っており、未知データでの100%を示しません。検証も人物・記録単位の分割ではないため、未見人物への性能は確認できていません。');
    end
    function exportPages(out)
        if ~isfolder(out),mkdir(out);end
        previous=tabs.SelectedTab;
        cleanup=onCleanup(@()restoreTab(previous));
        for q=1:6,tabs.SelectedTab=pages(q);drawnow;exportapp(fig,fullfile(out,sprintf('row%03d_step%d.png',picker.Value,q)));end
        e=getExplanation();writetable(e.featureTable,fullfile(out,sprintf('row%03d_features.csv',picker.Value)),'Encoding','UTF-8');
        writetable(e.pairTable,fullfile(out,sprintf('row%03d_svm.csv',picker.Value)),'Encoding','UTF-8');
        writetable(e.treeTable,fullfile(out,sprintf('row%03d_tree.csv',picker.Value)),'Encoding','UTF-8');
        metadata=struct('model','proto_v10','modelSHA256',har.hashFile(fullfile(root,'model','proto_v10','trained.mat')), ...
            'datasetFingerprint',d.fingerprint,'row',picker.Value,'truth',d.testLabels(picker.Value), ...
            'primary',e.primary,'votes',e.votes,'final',e.final,'voteApplied',e.voteApplied, ...
            'developmentCorrect',devMetric.correct,'developmentTotal',devMetric.total, ...
            'validationCorrect',valMetric.correct,'validationTotal',valMetric.total, ...
            'evaluationRole','development_tuned','images','MATLAB exportapp; no model retraining');
        har.writeJson(fullfile(out,sprintf('row%03d_snapshot.json',picker.Value)),metadata);
        fprintf('DEMO_EXPORTED: row %d -> %s\n',picker.Value,out);
    end
    function restoreTab(tab),if isvalid(fig),tabs.SelectedTab=tab;end,end
end

function g=pageGrid(parent,heights)
g=uigridlayout(parent,[numel(heights) 1]);g.RowHeight=heights;g.Padding=[16 12 16 12];g.RowSpacing=10;
end
function h=note(parent,text)
h=uilabel(parent,'Text',text,'WordWrap','on','FontSize',16,'VerticalAlignment','center','Interpreter','none');
end
function h=tableUI(parent,t,headers,widths)
values=table2cell(t);for k=1:numel(values),if isstring(values{k}),values{k}=char(values{k});end,end
h=uitable(parent,'Data',values,'ColumnName',headers,'ColumnWidth',widths,'RowName',{},'FontSize',14,'ColumnEditable',false);
end
function confusionPlot(parent,m,caption)
ax=uiaxes(parent);imagesc(ax,m.confusion);colormap(ax,flipud(gray));axis(ax,'square');
ax.XTick=1:6;ax.YTick=1:6;ax.XTickLabel=m.classes;ax.YTickLabel=m.classes;ax.XTickLabelRotation=30;
xlabel(ax,'予測');ylabel(ax,'正解');title(ax,caption,'FontSize',15);
for i=1:6,for j=1:6
    color=[0 0 0];if m.confusion(i,j)>max(m.confusion,[],'all')/2,color=[1 1 1];end
    text(ax,j,i,string(m.confusion(i,j)),'HorizontalAlignment','center','Color',color,'FontSize',15);
end,end
end
