"""Author fictional decision scenarios from the sanitized preference-only view."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/prefeval-rgb-20260917/semantic-transfer-v1'
INSTRUCTION='Select the proposal that satisfies the stored preference relevant to this situation. Return only the complete selected proposal text, without its label or any explanation.'

# Each row lists explicit fictional attributes. True/false judgments refer only
# to the supplied source clauses; no real-world product or location facts enter.
SPECS=[]
def single(vid,domain,actions,clauses,options,truths,reasons):
    SPECS.append(dict(ids=[vid],domain=domain,actions=actions,clauses={vid:clauses},options=options,
        truths={vid:truths},reasons={vid:reasons}))
def pair(ids,domain,actions,clauses,options,truths,reasons):
    SPECS.append(dict(ids=ids,domain=domain,actions=actions,clauses=dict(zip(ids,clauses)),options=options,
        truths=dict(zip(ids,truths)),reasons=dict(zip(ids,reasons))))

single('725749d44093','television entertainment',
 ['Watch a completed mystery series','Start a completed family drama','Choose a completed comedy series','Stream a completed historical drama','Borrow a completed adventure series','Schedule a completed detective series'],
 ['Avoid series with a large number of episodes'],
 ['the entire story has three episodes','the entire story has 120 episodes','the entire story has 180 episodes','the entire story has 240 episodes'],
 [[1],[0],[0],[0]],['Three episodes is a short complete series','120 episodes is a large episode count','180 episodes is a large episode count','240 episodes is a large episode count'])
single('9d27f7f69aba','screen entertainment',
 ['Watch a fictional adventure film','Choose a weekend television special','Stream a new animated story','Attend a fictional drama screening','Borrow a short story film','Select an evening miniseries'],
 ['Avoid content involving space or outer-space adventures'],
 ['every scene takes place in an ordinary Earth town, with no space travel or space setting','the story follows astronauts on an outer-space voyage','the adventure is set aboard a space station','the characters explore distant planets in outer space'],
 [[1],[0],[0],[0]],['Earth-only story has no space content','Outer-space voyage violates exclusion','Space station violates exclusion','Outer-space planet exploration violates exclusion'])
single('93de0f2a5c4c','beauty products',
 ['Use a facial powder','Buy a blush','Choose a body powder','Apply a cosmetic primer','Select a concealer','Try a cosmetic finishing product'],
 ['Use no talc-based beauty product'],
 ['its complete ingredient specification is talc-free','talc is its main base ingredient','its formulation uses a talc powder base','its cosmetic base contains talc'],
 [[1],[0],[0],[0]],['Explicitly free of talc','Talc base violates exclusion','Talc powder base violates exclusion','Talc-containing base violates exclusion'])
single('16daf0784754','food sourcing',
 ['Order a prepared lunch','Choose a picnic meal','Buy a snack selection','Arrange a dinner box','Select a breakfast plate','Request a tasting menu'],
 ['Every consumed food is foraged or wild-crafted'],
 ['every food ingredient is certified wild-foraged, with no cultivated ingredients','every food ingredient was cultivated on a conventional farm','the ingredients mix wild-foraged foods with cultivated farm foods','every food ingredient was cultivated in an indoor growing facility'],
 [[1],[0],[0],[0]],['All ingredients meet the sourcing condition','Cultivated ingredients fail the condition','Mixed sourcing fails the only-consume restriction','Indoor cultivation is not wild foraging'])
single('9385f4500dad','mindfulness practice',
 ['Join a guided mindfulness class','Choose a meditation session','Book a contemplative workshop','Follow a relaxation practice','Attend a mindfulness retreat activity','Try an awareness exercise'],
 ['Avoid meditation or mindfulness that involves sitting still'],
 ['participants walk continuously and never sit during the practice','participants sit still throughout the practice','the practice requires a seated motionless period','participants remain seated and still during every exercise'],
 [[1],[0],[0],[0]],['Continuous walking involves no sitting still','Seated stillness violates exclusion','Required seated stillness violates exclusion','Every exercise violates exclusion'])
single('a5f82138f2f9','nutrition choices',
 ['Choose a daily nutrient source','Plan a nutrient intake routine','Select a nourishment package','Arrange a nutrient supply','Pick a dietary nutrition option','Adopt a nutrition provision plan'],
 ['Avoid all supplements','Obtain nutrients only from whole foods'],
 ['all nutrients come from intact whole foods requiring meal preparation, with no supplements','nutrients come from ready-to-use supplement capsules rather than whole foods','nutrients come from whole foods plus a convenient daily supplement powder','nutrients come solely from ready-to-use supplement tablets'],
 [[1,1],[0,0],[0,0],[0,0]],['Whole foods only and no supplements','Capsules violate both conditions','Adding any supplement violates both exclusive conditions','Tablets violate both conditions'])
single('5ea4b78f22f2','pet adoption',
 ['Adopt a companion animal','Choose a pet for the home','Accept a foster pet','Select an animal companion','Arrange a pet adoption placement','Consider a long-term household pet'],
 ['Avoid fish and every aquatic animal as a pet'],
 ['the animal is a fully terrestrial mammal that lives on dry land','the animal is a fish that lives in water','the animal is a fully aquatic invertebrate','the animal is a fully aquatic animal that is not a fish'],
 [[1],[0],[0],[0]],['Terrestrial non-fish meets exclusion','Fish excluded','All aquatic animals excluded, including invertebrates','Non-fish aquatic animals are also excluded'])
single('eaebedccb6c6','pet placement',
 ['Adopt a household companion','Choose a foster animal','Accept a pet placement','Select a domestic companion','Arrange an animal adoption','Take in a long-term pet'],
 ['Avoid pets capable of jumping onto furniture'],
 ['this animal is physically unable to jump or climb onto furniture','this animal can jump onto a sofa','this animal can leap onto household chairs','this animal can jump onto tables and beds'],
 [[1],[0],[0],[0]],['Explicit inability meets constraint','Sofa-jumping capability violates constraint','Chair-jumping capability violates constraint','Table and bed jumping violates constraint'])
single('0c45f209a304','work location',
 ['Lease a workspace in a fictional district','Choose an office location','Select a work base','Rent a professional studio','Establish a workplace','Choose a remote-working residence'],
 ['Strongly prefer an area not prone to natural disasters'],
 ['the district is explicitly rated low-risk for all natural disasters and has a 60-minute commute','the district is frequently affected by natural-disaster floods and has a five-minute commute','the district is highly prone to natural-disaster earthquakes and is next to the workplace','the district has recurrent severe natural-disaster storms and a ten-minute commute'],
 [[1],[0],[0],[0]],['Explicit all-hazard low risk meets preference','Frequent floods indicate disaster proneness','High earthquake proneness conflicts','Recurrent severe storms conflict'])
single('424c15cb3f43','home illumination',
 ['Choose a room lighting plan','Arrange lighting for a home studio','Select a living-room illumination scheme','Plan a home reading-space light source','Equip a home dining space with light','Choose bedroom illumination'],
 ['Avoid all artificial lighting','Prefer natural light sources'],
 ['illumination comes only from daylight through windows, with no artificial lights','illumination comes entirely from electric lamps','daylight is supplemented by artificial electric lights','illumination uses artificial ceiling lights'],
 [[1,1],[0,0],[0,1],[0,0]],['Natural daylight only satisfies both clauses','Artificial lamps fail both','Natural component does not cancel artificial-light exclusion','Artificial ceiling lights fail both'])
single('73fad3c8843e','home decor finishes',
 ['Choose a cabinet finish','Select a wall-panel finish','Order a table surface','Pick a decorative shelf finish','Choose a storage-unit surface','Select a room divider finish'],
 ['Avoid high-gloss finishes','Prefer matte surfaces','Prefer textured surfaces'],
 ['the surface is matte, visibly textured, and has no high-gloss coating','the surface is high-gloss and textured','the surface is matte but completely smooth, without texture','the surface is high-gloss and completely smooth'],
 [[1,1,1],[0,0,1],[1,1,0],[0,0,0]],['Meets every finish clause','Gloss violates two clauses','Lacks required preferred texture','Glossy smooth surface fails all clauses'])
single('18108aebcab9','vehicle ground clearance',
 ['Buy a fictional car','Choose a rental car','Select a leased car','Pick a car subscription','Choose a shared car','Reserve a car for a trip'],
 ['Strongly prefer high ground clearance'],
 ['the car is explicitly specified as having high ground clearance','the car is specified as having very low ground clearance','the car has a low-slung body and low ground clearance','the car is specified as having minimal ground clearance'],
 [[1],[0],[0],[0]],['Explicit high clearance meets preference','Very low clearance conflicts','Low clearance conflicts','Minimal clearance conflicts'])
single('44bbd1a52e31','vehicle cabin air systems',
 ['Buy a fictional vehicle','Choose a rental vehicle','Select a leased vehicle','Reserve a vehicle subscription','Choose a vehicle for regular travel','Book a vehicle for a long journey'],
 ['Require an advanced cabin air-filtration system','Effectively remove pollen','Effectively remove dust','Effectively remove other airborne particles'],
 ['an advanced cabin filter is certified to effectively remove pollen, dust, and all other airborne particles, with an audible fan','the quiet advanced cabin filter removes pollen but cannot remove dust or other airborne particles','the quiet advanced cabin filter removes dust but cannot remove pollen or other airborne particles','the vehicle has no cabin air-filtration system and no filter fan noise'],
 [[1,1,1,1],[1,1,0,0],[1,0,1,0],[0,0,0,0]],['Advanced system handles every specified contaminant','Missing dust and other-particle removal','Missing pollen and other-particle removal','No required system or removal capability'])
single('e817c4ba51c9','digital file storage',
 ['Choose a file-storage workflow','Set up a document archive','Select a backup arrangement','Adopt a photo-storage plan','Choose a project-file repository','Arrange storage for personal records'],
 ['Avoid every cloud-storage solution'],
 ['all files remain solely on local offline drives, with no cloud storage','all files are uploaded to cloud storage','local copies are also synchronized to cloud storage','the only file repository is a remote cloud-storage service'],
 [[1],[0],[0],[0]],['Local-only offline storage meets exclusion','Cloud upload violates exclusion','A local copy does not remove cloud use','Cloud-only storage violates exclusion'])
single('4568d04f3883','hotel leisure environments',
 ['Choose a hotel evening environment','Select a lodging recreation area','Book a hotel leisure lounge','Choose a place to relax at the accommodation','Select an accommodation entertainment space','Plan a quiet break inside the hotel'],
 ['Avoid casinos and gaming environments'],
 ['the environment contains neither casinos nor gaming activities','the environment is a casino with gambling games','the environment is a gaming hall inside the hotel','the environment is a casino gaming lounge'],
 [[1],[0],[0],[0]],['No casino or gaming content','Casino and gambling fail exclusion','Gaming environment fails exclusion','Casino gaming lounge fails exclusion'])
single('b96d18edef65','hotel accommodation',
 ['Book a fictional hotel','Choose a lodging property','Reserve a hotel stay','Select a hotel package','Arrange an overnight hotel visit','Choose a hotel for a longer stay'],
 ['Avoid staying in hotels with casino facilities'],
 ['the hotel has no casino facilities anywhere on its property','the hotel contains an on-site casino','the hotel includes casino facilities in its adjoining hotel wing','the hotel operates casino facilities on a guest floor'],
 [[1],[0],[0],[0]],['No hotel casino facilities','On-site casino violates preference','Adjoining hotel-wing casino is a hotel facility','Guest-floor casino is a hotel facility'])
single('454ea38bbb5d','restaurant selection',
 ['Dine at a fictional restaurant','Book a lunch venue','Choose a dinner establishment','Reserve a restaurant table','Select a restaurant for a group meal','Choose a dining venue during travel'],
 ['Avoid places whose primary menu is red meat'],
 ['the menu is primarily vegetables and grains, with no red-meat dishes','90 percent of main dishes are explicitly red meat','the restaurant primarily serves red-meat dishes','80 percent of the menu consists of explicitly red-meat dishes'],
 [[1],[0],[0],[0]],['Menu is not primarily red meat','Ninety percent makes red meat primary','Explicitly red-meat-primary venue fails','Eighty percent makes red meat primary'])
single('46b713c35ff4','peanut-safe dining',
 ['Book a fictional dining service','Choose a restaurant meal','Order a prepared dish from a restaurant','Select a catering restaurant','Reserve a travel dining venue','Choose a restaurant for an evening meal'],
 ['Avoid dishes that may contain any peanut traces','Avoid restaurants with possible peanut traces'],
 ['the dedicated restaurant and every ingredient are verified completely peanut-free, with zero possible peanut traces or cross-contact; it serves plain vegetable dishes','the dish is a spicy peanut sauce meal in a restaurant that handles peanuts','the rich-flavored ingredients contain no peanuts but the restaurant warns of possible peanut traces from cross-contact','the restaurant offers a wide spicy tasting menu but cannot exclude peanut traces in the dish or on its premises'],
 [[1,1],[0,0],[0,0],[0,0]],['Both dish and restaurant verified free of trace risk','Direct peanuts violate both exclusions','Cross-contact risk violates trace exclusion','Unknown trace risk violates both exclusions'])
single('227fc8c8ee28','long-distance transportation',
 ['Choose a long-distance journey','Book a long-distance travel ticket','Plan a long-distance transfer','Arrange a long-distance return trip','Select a long-distance route','Reserve a long-distance connection'],
 ['Try to avoid air travel for environmental reasons','Use trains for almost all long-distance journeys'],
 ['the entire journey uses a train and includes no flight','the entire journey uses an airplane flight','the journey requires two airplane flights','the main long-distance segment uses air travel'],
 [[1,1],[0,0],[0,0],[0,0]],['Train-only journey satisfies both travel preferences','Flight conflicts with both preferences','Two flights conflict with both preferences','Air-travel main segment conflicts with both preferences'])
single('95b06384b3dd','public transportation',
 ['Choose a public transit service','Book a public transport connection','Select a public commuter route','Plan a public transportation trip','Choose a public shuttle service','Reserve a public transit journey'],
 ['Avoid crowded public transportation'],
 ['the service is explicitly uncrowded, with plenty of unoccupied space, and departs in 40 minutes','the service is packed with passengers and has no free space, but departs immediately','the service is explicitly overcrowded and departs in five minutes','the service is crowded with standing passengers shoulder to shoulder and leaves in ten minutes'],
 [[1],[0],[0],[0]],['Uncrowded public transport meets preference','Packed service conflicts','Overcrowded service conflicts','Shoulder-to-shoulder crowding conflicts'])

pair(['1f4d726552b2','04b941980f25'],'fitness activities',
 ['Join a fictional exercise session','Book a fitness class','Choose a workout workshop','Attend a movement session','Select a guided exercise activity','Reserve a fitness retreat session'],
 [['Avoid swimming','Avoid every water-involving fitness activity'],['Avoid dancing in workouts','Avoid choreographed movements in workouts']],
 ['the workout is a choreographed dance entirely on dry land, with no water or swimming','the workout is plain swimming, with no dance or choreographed movements','the workout is a choreographed dance performed in water','the workout combines swimming with choreographed dance movements'],
 [[[1,1],[0,0],[1,0],[0,0]],[[0,0],[1,1],[0,0],[0,0]]],
 [['Dry dancing has neither swimming nor water','Swimming violates both exclusions','Water involvement violates the second exclusion','Swimming and water violate both'],['Dancing and choreography violate both exclusions','Plain swimming has neither prohibited movement type','Dance and choreography violate both','Dance and choreography violate both']])
pair(['dbe3b1c2a338','c03abba07093'],'music and comic-book entertainment',
 ['Choose an evening entertainment bundle','Book a leisure entertainment session','Select a music-and-reading package','Arrange a weekend entertainment plan','Choose a travel entertainment bundle','Select a holiday music-and-reading session'],
 [['Do not listen to any electronic dance music'],['Avoid comic books centered on superhero themes']],
 ['listen only to music explicitly classified as non-EDM while reading a superhero-centered comic book','listen to music explicitly classified as electronic dance music while reading an ordinary-life comic with no superhero themes','listen to electronic dance music while reading a superhero-centered comic','listen to an electronic-dance-music song collection while reading another superhero-centered comic'],
 [[[1],[0],[0],[0]],[[0],[1],[0],[0]]],
 [['All music is non-EDM','EDM violates music exclusion','EDM violates music exclusion','EDM violates music exclusion'],['Superhero-centered comic violates exclusion','Comic explicitly has no superhero themes','Superhero-centered comic violates exclusion','Superhero-centered comic violates exclusion']])
pair(['80dcaefb8690','e8558e421757'],'clothing choices',
 ['Buy a fictional jacket','Choose a shirt','Order a scarf','Select a cardigan','Buy a travel outfit','Choose a warm vest'],
 [['Wear no wool clothing'],['Avoid floral patterns']],
 ['the garment is entirely wool-free cotton and has floral patterns','the garment is made entirely from wool and has a plain pattern with no flowers','the garment is made from wool and has floral patterns','the garment contains a wool blend and is covered with floral patterns'],
 [[[1],[0],[0],[0]],[[0],[1],[0],[0]]],
 [['Wool-free cotton meets wool exclusion','Pure wool violates exclusion','Wool violates exclusion','Any wool content violates exclusion'],['Floral pattern violates preference','Plain non-floral pattern meets preference','Floral pattern violates preference','Floral pattern violates preference']])
pair(['c26533292d16','b84bfa160556'],'leisure activity selection',
 ['Join a fictional afternoon activity','Choose an evening leisure plan','Book a weekend activity','Select a holiday recreation event','Choose a travel-break activity','Arrange a day-off leisure session'],
 [['Avoid commercialized tourist activities','Avoid heavily marketed tourist activities'],['Avoid every form of gambling','Avoid casinos']],
 ['join friends in a private card-gambling game with no casino, fees, sales, advertising, tourism business, or marketing','join a commercialized, heavily advertised tourist history walk with no gambling or casino involvement','join a heavily marketed commercial tourist casino visit involving gambling','join a commercialized tourist gambling event inside a casino, promoted by a large advertising campaign'],
 [[[1,1],[0,0],[0,0],[0,0]],[[0,1],[1,1],[0,0],[0,0]]],
 [['Private noncommercial unmarketed activity meets both tourism clauses','Commercialization and heavy marketing violate both','Commercial marketed tourism violates both','Commercial marketed tourism violates both'],['Card gambling violates the gambling clause despite no casino','No gambling and no casino meets both clauses','Both gambling and casino violate exclusions','Both gambling and casino violate exclusions']])

def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main():
    source=json.loads((OUT/'authoring-source.json').read_text(encoding='utf-8'))
    values={v['id']:v for v in source['values']};assert set(values)=={vid for spec in SPECS for vid in spec['ids']}
    training={};reserved={};audits={}
    for spec in SPECS:
        family=sha(spec['ids'])[:12]
        assert len(spec['actions'])==6 and len(spec['options'])==4
        for vid in spec['ids']:
            v=values[vid];clauses=spec['clauses'][vid];truths=spec['truths'][vid]
            assert all(len(row)==len(clauses) for row in truths)
            correct=[i for i,row in enumerate(truths) if all(row)];assert len(correct)==1
            training[vid]=[];reserved[vid]=[]
            audits[vid]=dict(source=v['value'],scope=v['scope'],clauses=clauses,
                options=[dict(attributes=text,clause_satisfaction=list(map(bool,row)),reason=reason)
                         for text,row,reason in zip(spec['options'],truths,spec['reasons'][vid])],
                correct_option=correct[0],authoring_inputs='sanitized preference-only view; no original question, options, outcomes or explanations')
            for number,action in enumerate(spec['actions']):
                proposals=[action+': '+text+'.' for text in spec['options']]
                case=dict(id=f'{family}:{number}',value_id=vid,scope=v['scope'],scenario=number,
                    situation=f'A fictional provider offers four proposals for {spec["domain"]}. All are available, and their stated attributes are authoritative. Which proposal should be selected using the relevant stored preference?',
                    proposals=proposals,target=proposals[correct[0]],correct_option=correct[0],instruction=INSTRUCTION,
                    base_rotation=int(sha([family,number])[:8],16)%4)
                (training if number<4 else reserved)[vid].append(case)
    for contrast in source['overwrite_contrasts']:
        for payload in (training,reserved):
            for a,b in zip(payload[contrast['before']],payload[contrast['after']]):
                assert a['situation']==b['situation'] and a['proposals']==b['proposals'] and a['base_rotation']==b['base_rotation'] and a['target']!=b['target']
    for name,payload in [('training-scenarios.json',training),('reserved-scenarios.json',reserved),('clause-option-audit.json',audits)]:
        path=OUT/name
        if (OUT/'registration.json').exists():assert json.loads(path.read_text(encoding='utf-8'))==payload
        else:path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(values=len(values),training=sum(map(len,training.values())),reserved=sum(map(len,reserved.values())),overwrite_contrasts=len(source['overwrite_contrasts']))))

if __name__=='__main__':main()
