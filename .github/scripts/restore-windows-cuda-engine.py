#!/usr/bin/env python3
"""Reuse released CUDA inference dependencies only when all source/build inputs match."""
import hashlib,json,os,re,sys,tempfile,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=Path(__file__).with_name('windows-cuda-engine-cache.json')
def digest(data):return hashlib.sha256(data).hexdigest()
def step_script(text,name):
 start=text.index('      - name: '+name+'\n')
 end=re.search(r'^      - (?:name:|uses:)',text[start+1:],re.M)
 block=text[start:start+1+end.start()] if end else text[start:]
 return block.split('        run: |\n',1)[1]
def output(hit):
 if os.environ.get('GITHUB_OUTPUT'):
  with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('hit='+str(hit).lower()+'\n')
def main():
 cache=json.loads(MANIFEST.read_text(encoding='utf-8'));workflow=(ROOT/'.github/workflows/windows.yml').read_text(encoding='utf-8')
 mismatches=[]
 for path,expected in cache['source_sha256'].items():
  if digest((ROOT/path).read_text(encoding='utf-8').encode())!=expected:mismatches.append(path)
 for name,expected in cache['recipe_sha256'].items():
  try:actual=digest(step_script(workflow,name).encode())
  except (ValueError,IndexError):actual='missing'
  if actual!=expected:mismatches.append(name)
 # Bind the CUDA installer and runner platform as well as both complete CMake commands.
 nvidia=workflow.split('  windows-cuda-pom:',1)[1]
 for literal in cache['required_workflow_literals']:
  matches=literal in nvidia if '\n' in literal else literal in [line.strip() for line in nvidia.splitlines()]
  if not matches:mismatches.append(literal)
 if mismatches:
  print('CUDA inference cache miss: '+', '.join(mismatches));output(False);return 1 if '--check-only' in sys.argv else 0
 print('CUDA_ENGINE_SOURCE_MATCH: '+cache['source_commit'])
 if '--check-only' in sys.argv:return 0
 directory=os.environ.get('RUNNER_TEMP') or tempfile.gettempdir()
 fd,name=tempfile.mkstemp(prefix='keryx-cuda-engine-',suffix='.zip',dir=directory);os.close(fd);archive=Path(name)
 try:
  h=hashlib.sha256()
  request=urllib.request.Request(cache['archive_url'],headers={'User-Agent':'Keryx-source-matched-dependency-cache'})
  with urllib.request.urlopen(request,timeout=180) as response,archive.open('wb') as dest:
   for block in iter(lambda:response.read(1024*1024),b''):h.update(block);dest.write(block)
  if h.hexdigest()!=cache['archive_sha256']:raise RuntimeError('Released dependency archive SHA256 mismatch')
  with zipfile.ZipFile(archive) as package:
   for filename,expected in cache['dll_sha256'].items():
    dest=ROOT/filename;temporary=dest.with_suffix('.dll.tmp');h=hashlib.sha256()
    with package.open(filename) as source,temporary.open('wb') as out:
     for block in iter(lambda:source.read(1024*1024),b''):h.update(block);out.write(block)
    if h.hexdigest()!=expected:temporary.unlink();raise RuntimeError('Released inference DLL SHA256 mismatch: '+filename)
    temporary.replace(dest)
    print('Verified unchanged inference dependency: '+filename+' '+expected)
  output(True)
 finally:archive.unlink(missing_ok=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
