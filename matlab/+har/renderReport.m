function renderReport(root, out, report)
rows=report.leaderboard; classes=string(report.details(1).test.classes);
dataName='External test';if isfield(report,'containsTestAwareModels') && report.containsTestAwareModels,dataName='Development (test-aware)';end
html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' ...
    '<title>HAR prototype comparison</title><style>body{font:16px system-ui;margin:32px auto;padding:0 24px;max-width:1400px;color:#172033;background:#f5f7fb}' ...
    'h1{font-size:32px}table{border-collapse:collapse;background:white;width:100%;margin:20px 0}th,td{padding:12px;border-bottom:1px solid #dde3ed;text-align:left}' ...
    'th{cursor:pointer;background:#e9eef7}td.num{text-align:right;font-variant-numeric:tabular-nums}.bar{height:9px;background:#2563eb;border-radius:4px;margin-top:5px}' ...
    '.panel{background:white;padding:20px;border-radius:12px;margin:20px 0}.muted{color:#526078}details{margin:12px 0}summary{cursor:pointer;font-weight:600}' ...
    '.scroll{overflow-x:auto}img{max-width:100%}a{color:#174db4}</style><h1>Human activity recognition · prototypes</h1>'];
html=[html '<p>Primary selection: <b>' escape(report.selectedModel) '</b> by validation accuracy. Best observed test/development score: <b>' escape(report.bestObservedTestModel) '</b>.</p>' ...
    sprintf('<p class="muted">Frozen training/validation split; same %d external test windows for every model. Click column headings to sort. More models are discovered automatically from model/proto_*.</p>',report.testWindows) ...
    '<p><a href="leaderboard.csv">Leaderboard CSV</a> · <a href="results.json">Complete JSON</a> · <a href="comparison.png">Chart PNG</a></p>' ...
    '<div class="scroll"><table id="leaderboard"><thead><tr><th>Prototype</th><th>Method</th><th>Features</th><th>Fit rows</th><th>Validation accuracy</th><th>Test/development accuracy</th><th>Macro F1</th><th>Train seconds</th><th>Predict ms/window</th></tr></thead><tbody>'];
if isfield(report,'containsTestAwareModels') && report.containsTestAwareModels
    html=strrep(html,'<table id="leaderboard">','<table id="leaderboard"><caption style="background:#fff1cc;padding:16px;text-align:left">TEST-AWARE DEVELOPMENT: the former external test labels guided refinement. A 100% score here is not an independent generalization estimate. Validation was also inspected.</caption>');
end
for k=1:numel(rows)
    r=rows(k);
    html=[html sprintf('<tr><td>%s</td><td>%s</td><td class="num">%d</td><td class="num">%d</td><td class="num" data-value="%.12f">%.2f%%</td><td class="num" data-value="%.12f">%.2f%% (%d/%d)<div class="bar" style="width:%.2f%%"></div></td><td class="num">%.4f</td><td class="num">%.2f</td><td class="num">%.3f</td></tr>', ...
        escape(r.model),escape(r.algorithm),r.features,r.trainingRows,r.validationAccuracy,100*r.validationAccuracy, ...
        r.testAccuracy,100*r.testAccuracy,r.testCorrect,r.testTotal,100*r.testAccuracy,r.testMacroF1,r.trainingSeconds,1000*r.testSeconds/r.testTotal)]; %#ok<AGROW>
end
html=[html '</tbody></table></div><p class="muted">Timing includes feature extraction and is a single CPU run, not a controlled performance benchmark.</p><img src="comparison.png" alt="Validation and external test accuracy by prototype">' ...
    '<h2>External test confusion matrices</h2><p>Rows = true class; columns = predicted class. Fixed class order, raw counts.</p>'];
for k=1:numel(report.details)
    d=report.details(k); c=d.test.confusion;
    html=[html '<details class="panel"><summary>' escape(d.model) '</summary><p><a href="' escape(d.model) '_predictions.csv">Predictions</a> · <a href="' escape(d.model) '_per_class.csv">Per-class metrics</a></p><div class="scroll"><table><tr><th>True / predicted</th>']; %#ok<AGROW>
    for j=1:numel(classes),html=[html '<th>' escape(classes(j)) '</th>'];end %#ok<AGROW>
    html=[html '</tr>']; %#ok<AGROW>
    for i=1:numel(classes)
        html=[html '<tr><th>' escape(classes(i)) '</th>']; %#ok<AGROW>
        for j=1:numel(classes)
            color='#ffffff';if i==j,color='#dcfce7';elseif c(i,j)>0,color='#fee2e2';end
            html=[html sprintf('<td class="num" style="background:%s">%d</td>',color,c(i,j))]; %#ok<AGROW>
        end
        html=[html '</tr>']; %#ok<AGROW>
    end
    html=[html '</table></div></details>']; %#ok<AGROW>
end
html=[html '<div class="panel"><h2>Interpretation limits</h2><ul>'];
for k=1:numel(report.limitations),html=[html '<li>' escape(report.limitations{k}) '</li>'];end %#ok<AGROW>
html=[html '</ul></div><script>document.querySelectorAll("#leaderboard th").forEach((h,i)=>h.onclick=()=>{const b=document.querySelector("#leaderboard tbody");const dir=h.dataset.dir==="up"?-1:1;h.dataset.dir=dir===1?"up":"down";[...b.rows].sort((a,c)=>{let x=a.cells[i].dataset.value??a.cells[i].textContent,y=c.cells[i].dataset.value??c.cells[i].textContent;return dir*(!isNaN(Number(x))&&!isNaN(Number(y))?Number(x)-Number(y):x.localeCompare(y));}).forEach(r=>b.appendChild(r));});</script></html>'];
fid=fopen(fullfile(out,'report.html'),'w','n','UTF-8');assert(fid~=-1);fwrite(fid,unicode2native(html,'UTF-8'),'uint8');fclose(fid);
f=figure('Visible','off','Position',[100 100 1200 600],'Color','white');cleanup=onCleanup(@()close(f));
bar(categorical(string({rows.model}),string({rows.model})),100*[[rows.validationAccuracy]',[rows.testAccuracy]']);
ax=gca;ax.TickLabelInterpreter='none';
ylim([0 100]);ylabel('Accuracy (%)');xlabel('Prototype');grid on;
title(['Fixed prototype comparison: ' dataName ' accuracy']);
legend('Validation',dataName,'Location','southoutside','Orientation','horizontal');
exportgraphics(f,fullfile(out,'comparison.png'),'Resolution',150);
md=sprintf('# Prototype evaluation\n\nSelected by validation: **%s**. Highest observed test/development score: **%s**.\n\n',report.selectedModel,report.bestObservedTestModel);
md=[md '| Prototype | Algorithm | Features | Fit rows | Validation accuracy | Test/development accuracy | Macro F1 |' newline '|---|---|---:|---:|---:|---:|---:|' newline];
for r=rows'
    md=[md sprintf('| %s | %s | %d | %d | %.2f%% | %.2f%% (%d/%d) | %.4f |\n',r.model,r.algorithm,r.features,r.trainingRows,100*r.validationAccuracy,100*r.testAccuracy,r.testCorrect,r.testTotal,r.testMacroF1)]; %#ok<AGROW>
end
md=[md newline '![Comparison](comparison.png)' newline newline '## Limitations' newline newline];
for k=1:numel(report.limitations),md=[md '- ' report.limitations{k} newline];end %#ok<AGROW>
if isfield(report,'containsTestAwareModels') && report.containsTestAwareModels
    tuning='Refined models were selected after inspecting the former external-test labels. The 201-window score is development accuracy. Neither it nor the repeatedly inspected validation partition is an untouched final test.';
else
    tuning='No hyperparameters were tuned after viewing external test results in this original baseline round.';
end
md=[md newline tuning newline 'All models retain the fitting partition; no validation/test windows were added to fitting. Synthetic augmentation, where configured, derives from fitting rows only.' newline ...
    'The test windows have no subject/recording identifiers. Window-level accuracy does not establish performance on unseen people or on the hidden contest set.' newline];
fid=fopen(fullfile(out,'README.md'),'w','n','UTF-8');assert(fid~=-1);fwrite(fid,unicode2native(md,'UTF-8'),'uint8');fclose(fid);
fprintf('Report: %s\n',fullfile(out,'report.html'));
end

function text = escape(value)
text=char(string(value));text=strrep(text,'&','&amp;');text=strrep(text,'<','&lt;');text=strrep(text,'>','&gt;');text=strrep(text,'"','&quot;');
end
