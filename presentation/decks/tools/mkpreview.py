import json,sys,os,re
root=sys.argv[1]; out=sys.argv[2]; img=sys.argv[3]
blobs={'97c7ca472058dfea0861a328a3d05b48':'final_confusion_model.png','ff9fbcca9ca9fc1c9e6081dbec685e34':'final_confusion_compare.png','242861f90dc085b2b7058bb28630161d':'final_confusion_model.png','08afd5e5d6ac05b163ce7e3e08d121be':'sit_stand_tilt.png','10ba6831cedda58d92418eb08bbaf43a':'sit_stand_gyro_sway.png','388d499385e388c3642abb5413eca94e':'sit_stand_pivot.png'}
d=json.load(open(os.path.join(root,'project/deck.json'),encoding='utf-8'))
css=open(os.path.join(root,'project/deck.css'),encoding='utf-8').read()
parts=[]
for sid in d['order']:
    h=open(os.path.join(root,'project/slides',sid+'.html'),encoding='utf-8').read()
    h=re.sub(r'<aside>.*?</aside>','',h,flags=re.S)
    for k,v in blobs.items(): h=h.replace('/_blob/'+k,img+'/'+v)
    parts.append('<div class="frame">'+h+'</div>')
html='''<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&display=swap"><style>
*{margin:0;padding:0;box-sizing:border-box} body{background:#888;counter-reset:page}
.frame{width:1920px;height:1080px;position:relative;overflow:hidden;margin:0 0 20px 0}
.frame>section{width:1920px;height:1080px;position:relative}
table{border-collapse:collapse} td,th{padding:10px 14px;text-align:left;border-bottom:1px solid #D3DCE8}
'''+css+'''</style></head><body>'''+''.join(parts)+'</body></html>'
open(out,'w',encoding='utf-8').write(html)
