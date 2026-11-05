"""Use the saved refinement checkpoint with the unchanged neutral prompt."""
import argparse
import uuid
from pathlib import Path
import research as r

def local_path(value,subtree):
    path=(r.ROOT/value).resolve()
    if not path.is_relative_to((r.ROOT/subtree).resolve()): raise ValueError('Path leaves experiment subtree')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt'); parser.add_argument('--model',default='models/zundamon-edited')
    parser.add_argument('--output',help='New log path inside continuation_refine/runs')
    args=parser.parse_args()
    folder=local_path(args.model,'models')
    output=local_path(args.output or f'runs/inference/{uuid.uuid4().hex}.json','runs')
    if output.exists(): raise FileExistsError(output)
    if not folder.is_dir(): raise FileNotFoundError(folder)
    model,tok=r.e.load_model(str(folder))
    result=r.prior.generate(model,tok,[{'id':'interactive','category':'user','prompt':args.prompt}],output)
    result['checkpoint']=folder.relative_to(r.ROOT).as_posix(); r.e.dump(output,result)
    print(result['responses'][0]['answer'])

if __name__=='__main__': main()
