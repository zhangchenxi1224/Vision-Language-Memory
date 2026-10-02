"""Recorded semantic edits to the 154 model-authored question drafts, before training."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from vision_memory.prefeval.official_ab import validate_forms, records, sha

# Reviewed against questions only; no preference labels, model answers or scores.
EDITS={
 'education_learning_styles:0026':dict(T2='Which convenient study methods can I use on-the-go?'),
 'entertain_games:0007':dict(T2='Which thrilling games would you recommend for me?',T3='Please suggest some thrilling games for me.'),
 'entertain_games:0020':dict(T2='Which new online games would you recommend that I play?'),
 'entertain_music_book:0016':dict(O1='I would like interesting audiobooks for my daily commute. Which titles should I listen to?'),
 'entertain_music_book:0006':dict(T2='Which acts or artists would you recommend as must-sees for the music festival I plan to attend this summer?'),
 'entertain_music_book:0043':dict(T3='As a big fan of live music hoping to attend festivals this summer, please recommend some music events for me.'),
 'entertain_shows:0023':dict(T3='Please recommend some captivating new TV shows for me to start watching.'),
 'entertain_shows:0037':dict(T2='Which of the best TV series would you recommend I begin watching this season?'),
 'entertain_sports:0009':dict(T2='As an adventurous visitor to Hawaii seeking thrills, which unique outdoor activities would you recommend for me?',T3='Please suggest unique outdoor activities that offer thrilling experiences for me, an adventurous person visiting Hawaii.',O1='I am adventurous and will be visiting Hawaii in search of thrilling experiences. Which unique outdoor activities would you recommend for me?'),
 'entertain_sports:0015':dict(T2='Which distinctive sports experiences would you recommend for my visit to Boston?',T3='Please recommend some unique sports experiences for my trip to Boston.'),
 'entertain_sports:0019':dict(T3='Please recommend unique sporting events or activities for my upcoming Texas visit that let me experience the local culture and traditions.'),
 'lifestyle_beauty:0014':dict(T3='Please recommend must-visit beauty boutiques or makeup counters for my first visit to Paris.',O1='This will be my first visit to Paris. Which beauty boutiques or makeup counters should I make a point of visiting?'),
 'lifestyle_dietary:0013':dict(T3='Please suggest some local dishes for me to try while traveling to Thailand.'),
 'lifestyle_dietary:0017':dict(O1='I will soon be in Thailand and am excited about its local cuisine. Which Thai dishes should I make a point of trying?'),
 'lifestyle_dietary:0025':dict(T2='Which dishes from Indian cuisine would you recommend that I try?',T3='Please recommend some Indian dishes for me to try.',O1='I would like to try Indian cuisine. Which dishes would you recommend for me?'),
 'lifestyle_dietary:0031':dict(O1='My visit to Italy is coming up soon. Which local dishes would you recommend that I try there?'),
 'lifestyle_dietary:0050':dict(T3='Please recommend authentic local Italian dishes for me to savor during my trip to Italy next month.'),
 'lifestyle_fit:0000':dict(T3='Please recommend a new workout to help me build endurance and cardiovascular fitness.'),
 'lifestyle_fit:0021':dict(T2='Which types of high-intensity interval training (HIIT) workouts would you suggest I add to my exercise regimen?',T3='Please recommend types of high-intensity interval training (HIIT) workouts for me to add to my exercise regimen.'),
 'lifestyle_health:0042':dict(T3='Please suggest effective ways for me to monitor and manage my physical health.'),
 'pet_ownership:0016':dict(T3='Please recommend some small pets that are easy for me to care for.'),
 'pet_ownership:0031':dict(T2='Which new pets would you recommend for my spouse and me to enjoy together and bond over?',T3='Please suggest a new pet that my spouse and I can both enjoy and bond over.'),
 'pet_ownership:0038':dict(O1='I need a pet suitable for an apartment that requires little maintenance. What options would you recommend?'),
 'professional_work_location_style:0034':dict(T2='Which engaging team-building activities would you recommend for my company to organize?'),
 'shop_fashion:0025':dict(O1='As a woman, I need new tops for the warm weather of the approaching summer. Which styles would you recommend for me?'),
 'shop_home:0042':dict(O1='I want to bring life to my rather dull living room. How would you suggest adding visual interest and personality?'),
 'travel_activities:0051':dict(T2='Which attractions or experiences would you recommend for my first trip to Italy next month?'),
 'travel_activities:0035':dict(O1='I want to explore Shanghai and am considering what to do there. Which great activities would you suggest for me?'),
 'travel_activities:0055':dict(O1='I am planning what to see and do on my Paris trip. Which places and activities should I make sure to include?'),
 'travel_hotel:0027':dict(O1='I need a hotel for my vacation in Cancun. Which options would you recommend as the best?'),
 'travel_hotel:0049':dict(O1='My trip to Los Angeles is coming up and I need somewhere to stay. Which great accommodation options would you suggest?'),
 'travel_restaurant:0008':dict(T2='Which excellent restaurants would you recommend that I try in Bordeaux?'),
 'travel_restaurant:0044':dict(T3='Please recommend authentic Thai restaurants in Bangkok that would suit me for dining.'),
 'travel_restaurant:0054':dict(T2='Which trendy places to dine would you recommend in Miami Beach?'),
}

def main():
    draft=Path(sys.argv[1]);out=Path(sys.argv[2]);rs=records(ROOT/'reports/prefeval-official-alignment-20260923')
    bank={};changes=[]
    for r in rs:
        raw=json.loads((draft/(r['id'].replace(':','-')+'.json')).read_text(encoding='utf-8'))
        original=raw['forms'];forms=dict(original)
        for k in ('T2','T3','O1'):
            text=forms[k]
            for old,new in [('your ','my '),("You're ","I'm "),('You’re ','I’m '),('You are ','I am '),
                            ('you are looking','I am looking'),("you're looking","I'm looking"),
                            ('you’re looking','I’m looking'),('you want to','I want to'),('You want to','I want to'),
                            ('should you ','should I '),('can you do','can I do'),('can you engage','can I engage'),
                            ('you can keep','I can keep')]:
                text=text.replace(old,new)
            forms[k]=text
        forms.update(EDITS.get(r['id'],{}))
        if '?' not in forms['O1']:
            forms['O1']=forms['O1'].rstrip('. ') + '. What would you suggest for me?'
        # Recast the already reviewed request rather than switching to the
        # assistant's own tastes or an unspecified third person's preferences.
        request=re.sub(r'^Please\s+','',forms['T3'],flags=re.I).rstrip('. ')
        request=request[0].lower()+request[1:]
        if request.startswith('as a big fan'):
            forms['O2']='As a big fan of live music hoping to attend festivals this summer, I would appreciate your recommendations for music events. What would you suggest for me?'
        else:
            forms['O2']=f'If I asked you to {request}, how would you respond?'
        validate_forms(r['question'],forms)
        normalized=[''.join(re.findall(r'[a-z0-9]+',q.lower())) for q in forms.values()]
        if len(set(normalized))!=5: raise ValueError('Punctuation-only paraphrase: '+r['id'])
        for k in forms:
            if forms[k]!=original[k]: changes.append(dict(id=r['id'],family=k,before=original[k],after=forms[k]))
        bank[r['id']]=dict(split=r['split'],forms=forms,draft_sha256=sha(draft/(r['id'].replace(':','-')+'.json')))
    payload=dict(status='semantic_reviewed_before_training',records=bank,
        author='Frozen Qwen3-VL-4B; question-only input; greedy generation',
        review='Codex reviewed original and rewritten questions; repaired task drift, pronoun role reversal, duplicate questions and lost constraints before any latent training',
        ood='O1 situation/question; O2 indirect request; not exposed to latent/FM training',
        changes=changes)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(records=len(bank),edits=len(changes),sha256=sha(out))))

if __name__=='__main__':main()
