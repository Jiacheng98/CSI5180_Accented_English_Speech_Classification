# "https://cloud.google.com/speech-to-text/docs/basics#overview"
# STEP 1: Downloaded Google Cloud (GC) SDK.
# STEP 2: Make sure the shell is auth'ed with the GC account 
#   2.1: gcloud auth login --no-browser
#   2.2: gcloud config set project csi5180project

import argparse
import json
import subprocess

import jiwer
from jiwer import wer
import num2words

def main():
    alternatives = call_google_API(f"accent_example/{args.fn}.wav", accent = args.accent)
    
    ground_truth_text = f"accent_example/{args.fn}.txt"
    with open(ground_truth_text) as f:
        gt_text = f.readlines()

    gt_text = gt_text[0].strip()
    predict_text = alternatives[0]['transcript']
    result_file = f"accent_example/{args.fn}-accentCode-{args.accent}"
    result_file = open(result_file, "w+")
    result_file.write(f"Ground truth text: \n{gt_text}\n\nPredicted text: \n{predict_text}")


    def convert_num_to_words(utterance):
          # utterance = ' '.join([num2words.num2words(i) if i.isdigit() else i for i in utterance.split()])
          utterance = [num2words.num2words(i) if i.isdigit() else i for i in utterance.split()]
          if "fifty-six" in utterance:
              number_index = utterance.index("fifty-six")
              utterance[number_index] = '56'
          utterance = ' '.join(utterance)
          return utterance

    predict_text = convert_num_to_words(predict_text)


    transformation_wer = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        # can be used to filter out white space.
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        # filter out multiple spaces between words.
        jiwer.RemoveMultipleSpaces(),
        # transform multiple sentences into a a single sentence
        jiwer.ReduceToSingleSentence(),
        # remove all leading and trailing spaces.
        jiwer.Strip(),
        # transform one or more sentences into a list of lists of words
        # this operation should be the final step of any transformation pipeline as the library internally computes the word error rate based on a double list of words.
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ]) 


    wer = jiwer.wer(
                    gt_text, 
                    predict_text, 
                    truth_transform=transformation_wer, 
                    hypothesis_transform=transformation_wer)

    # for demo
    transformation_demo = jiwer.Compose([
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        # can be used to filter out white space.
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        # filter out multiple spaces between words.
        jiwer.RemoveMultipleSpaces(),
        # transform multiple sentences into a a single sentence
        jiwer.ReduceToSingleSentence(),
        # remove all leading and trailing spaces.
        jiwer.Strip()
    ]) 


    gt_text = transformation_demo(gt_text)
    predict_text = transformation_demo(predict_text)
    result_file.write(f"\n\nGround truth text After Preprocessing: \n{gt_text}\n\nPredicted text After Preprocessing: \n{predict_text}")
    result_file.write(f"\n\nWord Error Rate: {wer}")
    result_file.close()


def gc_script_runner(gc_script):
    qurry_result = subprocess.run(gc_script, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    qurry_out = qurry_result.stdout.decode("utf-8") 
    qurry_err = qurry_result.stderr.decode("utf-8")
    return qurry_out


def call_google_API(wav_file_path, accent, maxAlternatives = 1):
    script = ["gcloud",
                "ml",
                "speech",
                "recognize",
                f"{wav_file_path}",
                f"--max-alternatives={maxAlternatives}",
                f"--language-code={accent}"]
    qurry_result = gc_script_runner(script)
    json_result = json.loads(qurry_result)
    
    # return the alternatives list returned by the Speech-to-Text API 
    return json_result['results'][0]['alternatives']



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-fn", help = "filename, which is avaialbe in 'accent_example/' folder", default="US")
    parser.add_argument("-accent", help = "languageCode which is available in Google API", default="en-AU")
    args = parser.parse_args()

    main()