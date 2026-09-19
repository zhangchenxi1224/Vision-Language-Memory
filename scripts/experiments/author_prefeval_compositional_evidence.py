"""Plan11 primitive-fact authoring; reads sanitized preferences, never old options."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/prefeval-rgb-20260917'
OUT = REPORT / 'compositional-evidence-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Each pair supplies two independently relevant components. The truth values
# below are audit data only; rendered Reader questions contain factual values.
# No benchmark question, original answer, old option, or model failure is read.
SPECS = {
 'c03abba07093': ('comic content', ['first comic', 'second comic'],
   ['a family memoir with no superheroes', 'a gardening diary with no superheroes'],
   ['a story centered on a costumed superhero', 'a story centered on a superpowered hero']),
 'dbe3b1c2a338': ('music programming', ['opening recording', 'closing recording'],
   ['unamplified chamber music; genre: classical, not EDM', 'solo acoustic folk; genre: folk, not EDM'],
   ['genre: electronic dance music', 'genre: electronic dance music']),
 '725749d44093': ('series episode commitment', ['main-story release', 'required conclusion release'],
   ['2 episodes', '1 episode'], ['70 episodes', '60 episodes']),
 '9d27f7f69aba': ('screening program', ['first feature', 'second feature'],
   ['a courtroom story entirely on Earth with no space scenes', 'a village story entirely on Earth with no space scenes'],
   ['an expedition outside Earth into outer space', 'an adventure aboard an interplanetary spacecraft']),
 '93de0f2a5c4c': ('cosmetic kit', ['base product ingredient list', 'finishing product ingredient list'],
   ['rice starch and mica only', 'kaolin and iron oxide only'],
   ['talc and mica', 'talc and iron oxide']),
 '16daf0784754': ('meal sourcing', ['first ingredient source', 'second ingredient source'],
   ['berries gathered from uncultivated wild plants', 'mushrooms gathered in an unmanaged wild forest'],
   ['berries grown on a cultivated farm', 'mushrooms grown in a commercial growing house']),
 '04b941980f25': ('exercise movement', ['first activity stage', 'second activity stage'],
   ['self-paced movement without dance or prescribed choreography', 'independent repetitions without dance or choreography'],
   ['dance following a choreographed sequence', 'synchronized movements following choreography']),
 '1f4d726552b2': ('exercise environment', ['first activity stage location', 'second activity stage location'],
   ['an indoor floor; the activity uses no water', 'an outdoor dry track; the activity uses no water'],
   ['a pool with swimming required', 'a water basin with wading required']),
 '9385f4500dad': ('wellbeing practice', ['first mindfulness stage', 'second mindfulness stage'],
   ['continuous walking with no seated interval', 'continuous gentle movement with no seated interval'],
   ['sitting motionless for the practice', 'remaining seated and still for the practice']),
 'a5f82138f2f9': ('daily nutrient plan', ['morning nutrient source', 'evening nutrient source'],
   ['an intact apple and cooked beans, eaten as whole foods', 'whole vegetables and grains, eaten as foods'],
   ['a nutritional supplement capsule', 'a powdered nutrient supplement']),
 '5ea4b78f22f2': ('pet placement', ['first animal habitat', 'second animal habitat'],
   ['a terrestrial species living entirely on land; not a fish', 'a terrestrial species living entirely on land; not a fish'],
   ['a fish living in an aquarium', 'an aquatic species living in a tank of water']),
 'eaebedccb6c6': ('pet placement', ['first animal movement capability', 'second animal movement capability'],
   ['physically unable to jump onto furniture', 'physically unable to jump onto furniture'],
   ['able to jump onto household furniture', 'able to jump onto household furniture']),
 '0c45f209a304': ('work relocation', ['work-site hazard classification', 'residential-site hazard classification'],
   ['the complete all-natural-hazards map classifies this site as not disaster-prone', 'the complete all-natural-hazards map classifies this site as not disaster-prone'],
   ['the all-natural-hazards map classifies this site as flood-prone', 'the all-natural-hazards map classifies this site as earthquake-prone']),
 '80dcaefb8690': ('garment materials', ['outer shell and cuffs', 'inner lining and collar'],
   ['cotton and linen only', 'silk and cotton only'],
   ['cotton shell with wool cuffs', 'wool lining with cotton collar']),
 'e8558e421757': ('garment decoration', ['outer decoration', 'lining decoration'],
   ['plain solid color with no floral pattern', 'geometric stripes with no floral pattern'],
   ['a repeating flower pattern', 'a floral print']),
 '424c15cb3f43': ('room illumination', ['daytime light path', 'evening-use light path'],
   ['sunlight through a window, with no artificial emitter', 'remaining daylight through a skylight, with no artificial emitter'],
   ['an electrically powered lamp', 'an electrically powered LED strip']),
 '73fad3c8843e': ('home surface selection', ['table surface', 'cabinet surface'],
   ['matte finish with a visibly textured surface', 'matte finish with a rough textured surface'],
   ['a polished high-gloss smooth finish', 'a high-gloss smooth lacquer']),
 '18108aebcab9': ('vehicle clearance', ['clearance under front structure', 'clearance under rear structure'],
   ['28 cm', '27 cm'], ['9 cm', '8 cm']),
 '44bbd1a52e31': ('vehicle cabin intake', ['primary air intake', 'secondary air intake'],
   ['an advanced filtration unit effectively removes pollen, dust, and other airborne particles before cabin entry', 'an advanced filtration unit effectively removes pollen, dust, and other airborne particles before cabin entry'],
   ['an unfiltered opening admits pollen, dust, and other airborne particles to the cabin', 'an unfiltered bypass admits pollen, dust, and other airborne particles to the cabin']),
 'e817c4ba51c9': ('document workflow', ['primary file location', 'automatic backup destination'],
   ['an offline disk physically inside the local computer', 'a second offline disk physically inside the same room'],
   ['a remote cloud storage account', 'a remote cloud storage service']),
 'b84bfa160556': ('holiday activity schedule', ['morning activity', 'afternoon activity'],
   ['a guided walk with no gambling or casino visit', 'a craft session with no gambling or casino visit'],
   ['gambling in a casino', 'a casino gambling session']),
 'c26533292d16': ('holiday activity operation', ['morning activity operator', 'afternoon activity operator'],
   ['a local noncommercial cooperative; no tourist advertising or marketing', 'a volunteer community group; noncommercial and not marketed to tourists'],
   ['a commercial tourist attraction promoted by a mass-market advertising campaign', 'a commercial tour promoted heavily through tourist marketing']),
 '4568d04f3883': ('hotel stay itinerary', ['first overnight property', 'second overnight property'],
   ['a lodging property without casinos or gaming environments', 'a lodging property without casinos or gaming environments'],
   ['a lodging property containing a casino and gaming hall', 'a lodging property containing a casino and gaming hall']),
 'b96d18edef65': ('hotel stay itinerary', ['first overnight property facilities', 'second overnight property facilities'],
   ['guest rooms and a garden only; no casino facilities', 'guest rooms and a library only; no casino facilities'],
   ['guest rooms and an on-site casino', 'guest rooms and an on-site casino']),
 '454ea38bbb5d': ('dining itinerary', ['first restaurant menu focus', 'second restaurant menu focus'],
   ['primarily vegetables and grains; no red-meat dishes', 'primarily fish and vegetables; no red-meat dishes'],
   ['primarily beef and lamb dishes', 'primarily red-meat dishes']),
 '46b713c35ff4': ('restaurant food handling', ['dish ingredients', 'preparation and serving equipment'],
   ['the complete ingredient list excludes peanuts and peanut traces', 'dedicated peanut-free equipment; no peanut cross-contact or traces'],
   ['the dish includes peanut fragments', 'shared equipment leaves possible peanut traces']),
 '227fc8c8ee28': ('long-distance itinerary', ['first intercity leg', 'second intercity leg'],
   ['travel by train with no flight segment', 'travel by train with no flight segment'],
   ['travel by airplane', 'travel by airplane']),
 '95b06384b3dd': ('public transportation itinerary', ['outbound vehicle occupancy', 'return vehicle occupancy'],
   ['uncrowded, with ample unoccupied seats and personal space', 'uncrowded, with ample unoccupied seats and personal space'],
   ['crowded, with tightly packed standing passengers', 'crowded, with tightly packed standing passengers']),
}


def source_view():
    source = json.loads((REPORT / 'attribute-generalization-v1/authoring-source.json').read_text())
    return {'values': [{k: v[k] for k in ('id', 'scope', 'value')} for v in source['values']],
            'contrasts': [{k: c[k] for k in ('before', 'after', 'scope')} for c in source['overwrite_contrasts']]}


def rules(vid):
    if vid == '725749d44093':
        return 'Prefer the complete series with the fewest total required episodes; all other listed qualities are identical. No absolute episode threshold is introduced.'
    if vid == '18108aebcab9':
        return 'Ground clearance is limited by the lower of the two listed structures. Prefer the greatest resulting clearance; no minimum clearance threshold is introduced.'
    return 'Every listed component of the committed action must satisfy the complete source preference; no component may introduce its excluded condition.'


def build():
    source = source_view()
    values = {v['id']: v for v in source['values']}
    assert set(values) == set(SPECS)
    families = {vid: (vid,) for vid in values}
    for c in source['contrasts']:
        family = tuple(sorted((c['before'], c['after'])))
        families[c['before']] = families[c['after']] = family
    cases, audits = {}, {}
    for vid, value in values.items():
        family = families[vid]
        cases[vid], audits[vid] = [], []
        for situation in range(2):
            shared = digest(['compositional-evidence-11', family, situation])[:12]
            title = 'one indivisible two-part commitment' if situation == 0 else 'one fixed recurring package whose two parts are both mandatory'
            topics = ' and '.join(SPECS[f][0] for f in family)
            question = (f'A coordinator must choose {title} involving {topics}. '
                        'Every listed part will occur; no part may be omitted or substituted. '
                        'The following complete fictional specifications are authoritative. '
                        'Unlisted properties are identical across proposals. '
                        'Which complete proposal best fits the relevant stored preference?')
            if vid == '725749d44093':
                question += ' All listed releases belong to one series and are required to complete it.'
            if vid == '18108aebcab9':
                question += ' The lower of the two structural clearances determines the vehicle ground clearance.'
            for variant in range(2):
                facts, proposals, validity = [], [], {f: [] for f in family}
                for candidate in range(4):
                    rows = []
                    for feature_index, f in enumerate(family):
                        _, labels, good, bad = SPECS[f]
                        # Each factual edit changes concrete component values.
                        # The unique compatible identity moves by two places.
                        winner = feature_index + (2 if variant else 0)
                        flags = [True, True] if candidate == winner else ([False, True] if (candidate + variant) % 2 else [True, False])
                        if candidate == 3 and variant == 0:
                            flags = [False, False]
                        validity[f].append(all(flags))
                        for component in range(2):
                            label = labels[component]
                            if situation:
                                label = 'recurring package: ' + label
                            rows.append({'feature': f, 'component': label,
                                         'value': (good if flags[component] else bad)[component]})
                    facts.append(rows)
                    body = '; '.join(f"{r['component']}: {r['value']}" for r in rows)
                    proposals.append(f"Proposal {('North','East','South','West')[candidate]}: {body}.")
                gold = validity[vid].index(True)
                assert sum(validity[vid]) == 1
                case = dict(id=f'{shared}:{variant}', value_id=vid, scope=value['scope'],
                    scenario=2*situation+variant, situation=question, proposals=proposals,
                    target=proposals[gold], correct_option=gold,
                    instruction='Return only the complete selected proposal text without its label or an explanation.',
                    base_rotation=int(shared[:8],16)%4, situation_family=situation,
                    counterfactual_variant=variant, primitive_facts=facts)
                cases[vid].append(case)
                audits[vid].append(dict(case_id=case['id'], source_clause=value['value'],
                    rule=rules(vid), candidates=[dict(index=i, compatible=ok,
                    relevant_facts=[r for r in facts[i] if r['feature']==vid]) for i,ok in enumerate(validity[vid])],
                    correct_option=gold, primitive_fact_count_per_candidate=len(facts[0])))
    validate(source, cases, audits)
    return source, cases, audits


def validate(source, cases, audits):
    # Reconstruct compatibility from factual values, independently of stored flags.
    for vid, cc in cases.items():
        _, labels, good, bad = SPECS[vid]
        assert len(cc) == 4
        for c, audit in zip(cc, audits[vid]):
            compatible = []
            for rows in c['primitive_facts']:
                own = [r for r in rows if r['feature']==vid]
                assert len(own)==2
                compatible.append(all(r['value']==good[i] for i,r in enumerate(own)))
            assert sum(compatible)==1 and compatible.index(True)==c['correct_option']==audit['correct_option']
            assert c['target']==c['proposals'][c['correct_option']]
        for a,b in ((cc[0],cc[1]),(cc[2],cc[3])):
            assert a['situation']==b['situation'] and a['base_rotation']==b['base_rotation']
            assert a['correct_option']!=b['correct_option']
            aa=sorted(digest(rows) for rows in a['primitive_facts'])
            bb=sorted(digest(rows) for rows in b['primitive_facts'])
            assert aa!=bb, 'counterfactual must change facts, not permute complete specifications'
    for c in source['contrasts']:
        for a,b in zip(cases[c['before']],cases[c['after']]):
            for key in ('situation','proposals','base_rotation','id'):
                assert a[key]==b[key]
            assert a['correct_option']!=b['correct_option']
    assert len(cases)==28 and sum(map(len,cases.values()))==112


def main():
    source,cases,audits=build()
    OUT.mkdir(parents=True,exist_ok=True)
    for name, data in [('authoring-source.json',source),('scenarios.json',cases),('case-audits.json',audits)]:
        path=OUT/name
        if path.exists():
            assert json.loads(path.read_text())==data, 'frozen authoring output changed'
        else:
            path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(values=28,assignments=112,source_digest=digest(source),scenario_digest=digest(cases),
        provenance='sanitized preference clauses and contrast membership only; no old options, benchmark questions or model outputs')))


if __name__=='__main__':
    main()
