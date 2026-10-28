/** Narrow, backed-up Figure10 mirror; author with Artifact Tool, preserve OPC parts.
 * Run with bundled Node and --runtime-dir containing its node_modules junction.
 * The Python argument is the bundled document runtime (lxml/zipfile only).
 * Existing styles are reused; only the Deshpande worksheet payload may change.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, i, all) => {
  if (i % 2 === 0) pairs.push([value.replace(/^--/, ''), all[i+1]]);
  return pairs;
}, []));
for (const name of ['workbook', 'csv', 'audit', 'python', 'runtime-dir']) {
  if (!args[name]) throw new Error(`Missing --${name}`);
}
const require = createRequire(path.join(path.resolve(args['runtime-dir']), 'artifact-loader.cjs'));
const {Workbook, SpreadsheetFile, FileBlob} = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const audit = path.resolve(args.audit), previewDir = path.join(audit, 'benchmark');
await fs.mkdir(previewDir, {recursive:true});
const fragment = path.join(previewDir, 'fig10-authored-block.xlsx');
const receiptPath = path.join(audit, 'workbook-fig10-integration.json');
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const python = (code, parameters=[]) => {
  const result = spawnSync(args.python, ['-B', '-c', code, ...parameters], {encoding:'utf8'});
  if (result.status !== 0) throw new Error(result.stderr || result.stdout || 'Python verification failed');
  return JSON.parse(result.stdout);
};
const preflight = python(String.raw`
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import hashlib,json,sys,datetime,shutil
p,audit=map(Path,sys.argv[1:]);n={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile(p) as z:
 w=E.fromstring(z.read('xl/workbook.xml'));links={x.get('Id'):x.get('Target') for x in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
 sh=next(x for x in w.find('m:sheets',n) if x.get('name')=='Deshpande');rid=sh.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id');t=links[rid];target=t.lstrip('/') if t.startswith('/') else 'xl/'+t
 s=E.fromstring(z.read(target))
 for c in s.findall('.//m:sheetData/m:row/m:c',n):
  r=c.get('r');row=int(''.join(x for x in r if x.isdigit()));col=''.join(x for x in r if x.isalpha())
  if 50<=row<=73 and len(col)==1 and col<='L' and any(E.QName(x).localname in ('v','f','is') for x in c):raise ValueError('Target is occupied: '+r)
 if any(50<=int(r.get('r'))<=73 for r in s.findall('m:sheetData/m:row',n)):raise ValueError('Target rows already exist; review rather than replace')
 styles=E.fromstring(z.read('xl/styles.xml'));xfs=styles.find('m:cellXfs',n)
 for style in (21,28,34,35,36):
  if style>=len(xfs):raise ValueError('Expected existing audit styles unavailable')
 widths={str(i):8.43 for i in range(1,13)}
 for c in s.findall('m:cols/m:col',n):
  for i in range(max(1,int(c.get('min'))),min(12,int(c.get('max')))+1):widths[str(i)]=float(c.get('width'))
raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=audit/('workbook-before-fig10-'+stamp+'-'+digest[:8]+'.xlsx')
if backup.exists():raise ValueError('Refusing to overwrite backup')
shutil.copy2(p,backup)
assert backup.read_bytes()==raw
print(json.dumps({'before_sha256':digest,'before_bytes':len(raw),'backup':str(backup),'backup_sha256':digest,'sheet_path':target,'column_widths':widths,'status':'backed_up_not_yet_modified'}))
`, [path.resolve(args.workbook), audit]);
await fs.writeFile(receiptPath, JSON.stringify(preflight, null, 2));

// Read-only pre-edit preview; no import/export round trip of the original book.
const before = await SpreadsheetFile.importXlsx(await FileBlob.load(args.workbook));
const beforePng = await before.render({sheetName:'Deshpande', range:'A35:L43', scale:1.5, format:'png'});
await fs.writeFile(path.join(previewDir, 'deshpande-before-fig10.png'), new Uint8Array(await beforePng.arrayBuffer()));

const sourceCsv = await fs.readFile(args.csv, 'utf8');
const sourceBook = await Workbook.fromCSV(sourceCsv, {sheetName:'Source'});
const source = sourceBook.worksheets.getItem('Source').getUsedRange().values;
const names = source[0], records = source.slice(1).filter(r => r[0]).map(r => Object.fromEntries(names.map((k,i)=>[k,r[i]])));
if (records.length !== 18) throw new Error('Expected all18 initial/final digitised points');
const wb = Workbook.create(), sheet = wb.worksheets.add('Deshpande');
for (let i=1;i<=12;i++) sheet.getRange(`${String.fromCharCode(64+i)}1:${String.fromCharCode(64+i)}73`).format.columnWidth = preflight.column_widths[String(i)];
sheet.getRange('A50:L73').format.font = {name:'Arial',size:10};
sheet.getRange('A50:L73').format.verticalAlignment = 'center';
const titles = {
  50:'Hsieh et al. (2020): dimensional Figure 10 digitisation, 3 mm bars',
  51:'Source: Fig.10, published p.13; DOI 10.1016/j.jmps.2020.103925',
  52:'Graphical bounds: pixel sensitivity, not statistical confidence intervals. Initial and final K are distinct.',
  53:'Nominal rho follows manuscript: printed density positions are approximately tenfold lower than manuscript density.',
  73:'Canonical CSV: Z:/p1/sims/Ti/Validation/audit/benchmark/fig10-digitised.csv'
};
for (const [row,text] of Object.entries(titles)) {
  const range=sheet.getRange(`A${row}:L${row}`);range.merge();sheet.getRange(`A${row}`).values=[[text]];
  range.format.rowHeight=19;
}
sheet.getRange('A50:L50').format.font={name:'Calibri',size:11,bold:true};
const headers=['Topology','Stage','rho','l (mm)','Density text\n(kg/m3)','Density plot\n(kg/m3)','K\n(MPa sqrt(m))','K lower\n(graphical)','K upper\n(graphical)','Image x\n(pixel)','Image y\n(pixel)','Point ID'];
sheet.getRange('A54:L54').values=[headers];
sheet.getRange('A54:L54').format={fill:'#334155',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,horizontalAlignment:'center',verticalAlignment:'center',rowHeight:36};
const body=records.map(r => [r.topology,r.stage,Number(r.nominal_relative_density),Number(r.bar_length_mm),Number(r.manuscript_density_kg_m3),Number(r.printed_axis_density_kg_m3),Number(r.K_MPa_sqrt_m),Number(r.K_graphical_lower),Number(r.K_graphical_upper),Number(r.centre_x_pixel),Number(r.centre_y_pixel),r.point_id]);
sheet.getRange('A55:L72').values=body;
sheet.getRange('A55:L72').format.rowHeight=18;
sheet.getRange('C55:C72').setNumberFormat('0.00000');
sheet.getRange('F55:K72').setNumberFormat('#,##0.000');
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Deshpande!A54:L57',include:'values,formulas',tableMaxRows:4,tableMaxCols:12,maxChars:2500})).ndjson);
const authored=await SpreadsheetFile.exportXlsx(wb);await authored.save(fragment);
const fragmentSha=sha(await fs.readFile(fragment));

const applied = python(String.raw`
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json,sys,hashlib,copy,re,tempfile,os
original,fragment,receipt,tempdir=map(Path,sys.argv[1:]);state=json.loads(receipt.read_text());n={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};M='{'+n['m']+'}'
if hashlib.sha256(original.read_bytes()).hexdigest()!=state['before_sha256']:raise ValueError('Live workbook changed since backup')
with ZipFile(original) as zi,ZipFile(fragment) as fz:
 old={name:zi.read(name) for name in zi.namelist()};s=E.fromstring(old[state['sheet_path']]);f=E.fromstring(fz.read('xl/worksheets/sheet1.xml'));sst=[]
 if 'xl/sharedStrings.xml' in fz.namelist():sst=list(E.fromstring(fz.read('xl/sharedStrings.xml')))
 old_cells={c.get('r'):E.tostring(c,method='c14n') for c in s.findall('.//m:sheetData/m:row/m:c',n)};old_cols=E.tostring(s.find('m:cols',n),method='c14n');data=s.find('m:sheetData',n)
 for source_row in f.findall('m:sheetData/m:row',n):
  rownum=int(source_row.get('r'))
  if not 50<=rownum<=73:continue
  row=E.Element(M+'row',r=str(rownum))
  for key in ('ht','customHeight'):
   if key in source_row.attrib:row.set(key,source_row.get(key))
  for source_cell in source_row.findall('m:c',n):
   address=source_cell.get('r');letters=''.join(x for x in address if x.isalpha())
   if len(letters)!=1 or letters>'L':raise ValueError('Authored cell outside target: '+address)
   cell=copy.deepcopy(source_cell)
   if cell.get('t')=='s':
    value=int(cell.find('m:v',n).text);cell.remove(cell.find('m:v',n));cell.set('t','inlineStr');inline=E.SubElement(cell,M+'is')
    for child in sst[value]:inline.append(copy.deepcopy(child))
   # Reuse already verified existing styles; do not change styles.xml or SST.
   style=21 if rownum==50 else 34 if rownum==54 else 36 if 55<=rownum<=72 and letters=='C' else 35 if 55<=rownum<=72 and letters in 'FGHIJK' else 28
   cell.set('s',str(style));row.append(cell)
  data.append(row)
 merges=E.Element(M+'mergeCells')
 for merge in f.findall('m:mergeCells/m:mergeCell',n):merges.append(copy.deepcopy(merge))
 merges.set('count',str(len(merges)))
 later={'phoneticPr','conditionalFormatting','dataValidations','hyperlinks','printOptions','pageMargins','pageSetup','headerFooter','rowBreaks','colBreaks','customProperties','cellWatches','ignoredErrors','smartTags','drawing','legacyDrawing','legacyDrawingHF','picture','oleObjects','controls','webPublishItems','tableParts','extLst'}
 index=next((i for i,child in enumerate(s) if E.QName(child).localname in later),len(s));s.insert(index,merges)
 dimension=s.find('m:dimension',n);start,end=dimension.get('ref').split(':');column=''.join(c for c in end if c.isalpha());dimension.set('ref',start+':'+column+'73')
 payload=E.tostring(s,encoding='UTF-8',xml_declaration=True,standalone=True)
 fd,name=tempfile.mkstemp(suffix='.xlsx',prefix='fig10-candidate-',dir=tempdir);os.close(fd);candidate=Path(name)
 with ZipFile(candidate,'w') as zo:
  for info in zi.infolist():zo.writestr(info,payload if info.filename==state['sheet_path'] else old[info.filename])
with ZipFile(candidate) as z:
 if set(z.namelist())!=set(old):raise ValueError('Workbook package membership changed')
 changed=[name for name,raw in old.items() if z.read(name)!=raw]
 if changed!=[state['sheet_path']]:raise ValueError('Unexpected changed parts: '+repr(changed))
 after=E.fromstring(z.read(state['sheet_path']));after_cells={c.get('r'):E.tostring(c,method='c14n') for c in after.findall('.//m:sheetData/m:row/m:c',n)}
 for address,value in old_cells.items():
  if after_cells[address]!=value:raise ValueError('Original cell changed: '+address)
 if E.tostring(after.find('m:cols',n),method='c14n')!=old_cols:raise ValueError('Original column widths changed')
 added_cells=len(after_cells)-len(old_cells)
 if not 233<=added_cells<=288:raise ValueError('Unexpected authored-cell count in A50:L73')
 formulas=sum(len(E.fromstring(raw).findall('.//m:f',n)) for name,raw in old.items() if name.startswith('xl/worksheets/') and name.endswith('.xml'))
 charts=sum(name.startswith('xl/charts/') for name in old)
if hashlib.sha256(original.read_bytes()).hexdigest()!=state['before_sha256']:raise ValueError('Concurrent live workbook change; not replacing')
os.replace(candidate,original)
state.update(status='applied_and_package_verified',after_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),after_bytes=original.stat().st_size,changed_parts=changed,unchanged_existing_cells=len(old_cells),existing_formula_count_preserved=formulas,chart_parts_preserved=charts,package_membership_unchanged=True,column_widths_unchanged=True,original_styles_and_shared_strings_byte_identical=True,new_range='Deshpande!A50:L73',new_cells=added_cells)
receipt.write_text(json.dumps(state,indent=2));print(json.dumps(state))
`, [path.resolve(args.workbook), fragment, receiptPath, path.resolve(args['runtime-dir'])]);
const finalBook=await SpreadsheetFile.importXlsx(await FileBlob.load(args.workbook));
const finalBody=finalBook.worksheets.getItem('Deshpande').getRange('A55:L72').values;
for(let i=0;i<body.length;i++)for(let j=0;j<12;j++){
 const actual=finalBody[i][j],expected=body[i][j];
 if(typeof expected==='number' ? typeof actual!=='number'||Math.abs(actual-expected)>1e-10 : actual!==expected)
  throw new Error(`Read-back mismatch at row${55+i},column${1+j}`);
}
console.log((await finalBook.inspect({kind:'table',range:'Deshpande!A54:L57',include:'values,formulas',tableMaxRows:4,tableMaxCols:12,maxChars:2500})).ndjson);
const finalPng=await finalBook.render({sheetName:'Deshpande',range:'A50:L73',scale:1.8,format:'png'});
await fs.writeFile(path.join(previewDir,'deshpande-fig10.png'),new Uint8Array(await finalPng.arrayBuffer()));
await fs.unlink(fragment);
await fs.rm(fragment+'.inspect.ndjson',{force:true});
const receipt={...applied,source_csv:path.resolve(args.csv),source_csv_sha256:sha(Buffer.from(sourceCsv)),authored_fragment_sha256:fragmentSha,authored_fragment_retained:false,authoring:'Bundled Artifact Tool authored A50:L73; lossless OPC transplant reused existing styles',preview:path.join(previewDir,'deshpande-fig10.png')};
await fs.writeFile(receiptPath,JSON.stringify(receipt,null,2));
console.log(JSON.stringify(receipt));
