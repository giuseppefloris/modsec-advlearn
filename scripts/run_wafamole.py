import click
import json
import numpy as np
import random
import pickle
import re
import sqlglot
from wafamole.evasion import EvasionEngine
from wafamole.exceptions.models_exceptions import UnknownModelError
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.models import PyModSecurityWrapper
except ImportError:
    # ModSecurity module is not available
    pass
from sklearn_modsecurity_ml_waf import SklearnModSecurityMlWaf


def _load_json_conf(json_conf_path):
    try:
        with open(json_conf_path, 'r') as fp:
            json_conf = json.load(fp)
    except (json.JSONDecodeError, OSError) as e:
        raise Exception("Error loading the JSON config options for the target Web WAF:\n{}".format(e))
    else:
        return json_conf


@click.group()
def wafamole():
    pass


@wafamole.command()
@click.option("--waf-type", "-w", default="token", help="Target WAF", type=str)
@click.option("--timeout", "-t", default=14400, help="Timeout when evading the WAF", type=int)
@click.option(
    "--max-rounds", "-r", default=1000, help="Maximum number of fuzzing rounds", type=int
)
@click.option("--round-size", "-s", default=20, help="Fuzzing step size for each round (parallel fuzzing steps)", type=int
)
@click.option(
    "--threshold", default=0.5, help="Classification threshold of the target WAF (default 0.5)", type=float
)
@click.option(
    "--random-seed", default=0, help="Random seed to use (default: 0)", type=int
)
@click.option(
    "--output-path", "-o", default="output.json", help="Location were to save the adversarial examples found (JSON file).", type=str
)
@click.option(
    "--use-multiproc", "-m", default=False, help="Whether to enable multiprocessing for fuzzing", type=bool
)
@click.argument("waf-path", default="")
@click.argument("dataset-path")
def run_wafamole(
    waf_path,
    dataset_path,
    waf_type,
    timeout,
    max_rounds,
    round_size,
    threshold,
    random_seed,
    output_path,
    use_multiproc
):

    np.random.seed(random_seed)
    random.seed(random_seed)

    if re.match(r"modsecurity_pl[1-4]", waf_type):
        pl = int(waf_type[-1])
        try:
            waf = PyModSecurityWrapper(pl, rules_path=waf_path)
        except Exception as error:
            print("ModSecurity wrapper is not installed, see https://github.com/AvalZ/pymodsecurity to install")
            
    elif waf_type == "ml_model_crs":
        waf = SklearnModSecurityMlWaf(**_load_json_conf(waf_path))
    else:
        raise UnknownModelError("Unsupported WAF type: {}".format(waf_type))

    opt = EvasionEngine(waf, use_multiproc)

    try:
        with open(dataset_path, 'r') as fp:
            dataset = json.load(fp)
    except Exception as error:
        raise SystemExit("Error loading the dataset: {}".format(error))
    print("[INFO] Number of attack payloads: {}".format(len(dataset)))

    with open(output_path, 'w') as out_file:
        for sample in dataset[:3]:
            # sample_error, adv_sample_error = False, False
            # try:
            #     sqlglot.transpile(sample)
            # except sqlglot.errors.ParseError:
            #     sample_error = True

            best_score, adv_sample, scores_trace, _, _, _, _ = opt.evaluate(
                sample, 
                max_rounds, 
                round_size, 
                timeout, 
                threshold
            )

            # try:
            #     sqlglot.transpile(adv_sample)
            # except sqlglot.errors.ParseError:
            #     adv_sample_error = True

            # if adv_sample_error and sample_error:
            #     print("[ERROR] Both original payload and adv payload are not valid!\nPayload:{}\nAdv payload:{}\n".format(repr(sample), repr(adv_sample)))
            # elif adv_sample_error and not sample_error:
            #     print("[ERROR] Original payload is valid but adv payload is not valid!\nPayload:{}\nAdv payload:{}\n".format(repr(sample), repr(adv_sample)))
            # else:
            #     info = {'payload': sample, 'adv_payload': adv_sample, 'best_score': best_score}
            #     out_file.write(json.dumps(info) + '\n')

            info = {'payload': sample, 'adv_payload': adv_sample, 'best_score': best_score, 'scores_tarce': scores_trace}
            
            
            out_file.write(json.dumps(info) + '\n')


if __name__ == "__main__":
    run_wafamole()
