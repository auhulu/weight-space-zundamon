"""Run the saved edited model with the neutral system and a 128-token cap."""
import argparse
import experiment as e

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('prompt')
    p.add_argument('--model',default='models/zundamon-edited')
    a=p.parse_args()
    model,tok=e.load_model(str((e.ROOT/a.model).resolve()))
    result=e.generate(model,tok,[{'id':'interactive','category':'user','prompt':a.prompt}])
    print(result['responses'][0]['answer'])

if __name__=='__main__': main()
