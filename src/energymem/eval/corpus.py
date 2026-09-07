"""A real-text episodic corpus with known multi-target gold sets.

WHY THIS EXISTS. Every finding F1-F9 was measured on random Gaussian vectors, where
cosine sits at 1.000 on every condition (F6) and nothing can distinguish itself. That is
a debugging harness, not evidence. This module produces NATURAL LANGUAGE episodes with
the three properties that make retrieval genuinely hard:

  1. THREADS  -- storylines whose episodes are spread over time, so a question about a
                 storyline has SEVERAL correct answers (multi-target, the F4/F5 regime).
  2. CONFUSABLE PAIRS -- distinct threads that share vocabulary ("lease", "deposit",
                 "viewing" appear in both apartment-hunting and office-relocation), so
                 lexical and embedding similarity both produce real false positives.
  3. INDIRECTION -- questions phrased the way a person asks them, not the way the memory
                 is worded, so exact-match retrieval cannot win for free.

HONESTY. This is a purpose-built probe, not a public benchmark. It exists so we can find
out TODAY whether the synthetic findings survive real embeddings. LongMemEval and LoCoMo
remain the reportable benchmarks (docs/04_EXPERIMENT_PROTOCOL.md E6); nothing measured
here goes in the paper as a headline number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Episode:
    text: str
    thread: str
    t: int
    gold_for: list[str] = field(default_factory=list)


@dataclass
class Question:
    text: str
    gold_idx: list[int]
    thread: str
    kind: str


# ---------------------------------------------------------------- thread templates
# Each thread: (id, topic-family, [episode templates], [(question, which episodes)])
# Confusable families share a topic-family tag so we can plant near-miss distractors.

THREADS: list[dict] = [
    dict(
        id="apartment", fam="property",
        eps=[
            "Viewed the two-bedroom on Warren Street; the kitchen was tiny but the light was good.",
            "The broker wants a two-month deposit up front, which is more than I budgeted.",
            "Decided against Warren Street because the commute would be forty minutes each way.",
            "Saw a place on Cardigan Road that is smaller but ten minutes from the office.",
            "Signed the lease on Cardigan Road this afternoon; move-in is the fourteenth.",
            "Paid the deposit by transfer and the agent confirmed receipt.",
        ],
        qs=[("what did I end up deciding about the flat", [2, 3, 4], "decision"),
            ("how much was I asked to pay before moving in", [1, 5], "fact")],
    ),
    dict(
        id="office_move", fam="property",
        eps=[
            "The company is relocating; we toured the new floor near the canal today.",
            "Facilities said the lease on the current office runs out in March.",
            "The new space has fewer meeting rooms, which the design team objected to.",
            "We were told the move date is the first week of April.",
        ],
        qs=[("when are we changing offices", [1, 3], "fact")],
    ),
    dict(
        id="knee", fam="health",
        eps=[
            "The knee started hurting again after the Sunday run, mostly on stairs.",
            "The physio thinks it is the tendon rather than the joint itself.",
            "She gave me a set of loading exercises to do three times a week.",
            "Told to stop running entirely for six weeks, which is frustrating.",
            "Six weeks later the stairs are fine but squatting still aches.",
        ],
        qs=[("what did the physiotherapist actually tell me to do", [2, 3], "instruction"),
            ("how is the leg problem progressing", [0, 4], "trajectory")],
    ),
    dict(
        id="dentist", fam="health",
        eps=[
            "Chipped a molar on an olive stone at dinner.",
            "The dentist can only fit me in a fortnight from Thursday.",
            "Turned out to need a crown rather than a simple filling.",
        ],
        qs=[("what happened with my tooth", [0, 2], "fact")],
    ),
    dict(
        id="thesis", fam="work",
        eps=[
            "Supervisor said the introduction is doing too much and should be halved.",
            "She wants the evaluation chapter rewritten around one claim, not four.",
            "Agreed to send a revised outline before the end of the month.",
            "The revised outline came back with one comment: cut the survey section.",
            "Rewrote the evaluation around the reduction result and it reads far better.",
        ],
        qs=[("what feedback did I get on the writing", [0, 1, 3], "feedback"),
            ("what did I promise to send and when", [2], "commitment")],
    ),
    dict(
        id="grant", fam="work",
        eps=[
            "The department is putting together a funding application for next cycle.",
            "I was asked to write the methods section by the fifteenth.",
            "The budget was cut before submission so the travel line disappeared.",
        ],
        qs=[("what was I asked to contribute to the application", [1], "commitment")],
    ),
    dict(
        id="japan", fam="travel",
        eps=[
            "Booked flights to Osaka for the second week of November.",
            "Reserved a room in Kyoto for three nights in the middle of the trip.",
            "The rail pass has to be bought before arriving, not at the station.",
            "Changed the Kyoto nights to Nara because the trains work out better.",
            "Packed light because the November weather there is mild.",
        ],
        qs=[("where am I staying on that trip", [1, 3], "revision"),
            ("what do I need to sort out before I fly", [2], "todo")],
    ),
    dict(
        id="portugal", fam="travel",
        eps=[
            "Thinking about Lisbon in spring, mostly for the food.",
            "Flights are cheapest on Tuesdays according to the tracker.",
            "Decided to postpone until autumn because of the conference clash.",
        ],
        qs=[("what happened to the Lisbon plan", [2], "decision")],
    ),
    dict(
        id="car", fam="admin",
        eps=[
            "The warning light came on again on the motorway.",
            "Garage says it is the sensor, not the engine, which is a relief.",
            "The part has to be ordered and will take a week to arrive.",
            "Picked the car up on Friday and the light has stayed off since.",
        ],
        qs=[("what was actually wrong with the car", [1], "fact"),
            ("how did the garage situation resolve", [2, 3], "trajectory")],
    ),
    dict(
        id="insurance", fam="admin",
        eps=[
            "The home insurance renewal quote went up by a third.",
            "Called them and they matched a competitor after some pushing.",
            "The new policy excludes accidental damage, which I did not notice at first.",
        ],
        qs=[("what should I watch out for in the new policy", [2], "caution")],
    ),
    dict(
        id="sourdough", fam="food",
        eps=[
            "The starter has been sluggish since the kitchen got cold.",
            "Moved it to the airing cupboard and it doubled overnight.",
            "Tried a longer cold ferment and the crumb was much more open.",
            "Reduced the hydration slightly because the dough was unmanageable.",
        ],
        qs=[("what actually fixed the bread", [1, 2, 3], "trajectory")],
    ),
    dict(
        id="coffee", fam="food",
        eps=[
            "The grinder is producing too many fines and the shots run fast.",
            "Adjusted the burrs a notch finer and the timing improved.",
        ],
        qs=[("what did I change about the coffee setup", [1], "fact")],
    ),
    dict(
        id="guitar", fam="hobby",
        eps=[
            "Started learning the Bach prelude and the stretch in bar four is impossible.",
            "The teacher suggested an alternative fingering for that bar.",
            "Practised it slowly for a fortnight and it is now comfortable.",
        ],
        qs=[("how did I get past the difficult part", [1, 2], "trajectory")],
    ),
    dict(
        id="running", fam="hobby",
        eps=[
            "Signed up for the half marathon in the spring.",
            "The training plan has four runs a week which feels like a lot.",
            "Dropped to three runs a week after the knee trouble started.",
        ],
        qs=[("how did the training schedule change", [1, 2], "revision")],
    ),
    dict(
        id="laptop", fam="tech",
        eps=[
            "The laptop battery is now lasting under two hours.",
            "The repair shop quoted more than half the price of a new machine.",
            "Bought a refurbished one instead and moved everything across at the weekend.",
        ],
        qs=[("what did I do about the computer", [1, 2], "decision")],
    ),
    dict(
        id="backup", fam="tech",
        eps=[
            "The external drive failed to mount this morning.",
            "Most things were in the cloud but the photo archive was not.",
            "Set up an automatic weekly backup so this cannot happen again.",
        ],
        qs=[("what did I lose and what did I change afterwards", [1, 2], "trajectory")],
    ),
    dict(
        id="cat", fam="home",
        eps=[
            "The cat has stopped eating the dry food she used to like.",
            "The vet found nothing wrong and suggested trying a different brand.",
            "She eats the new food fine, so it was fussiness rather than illness.",
        ],
        qs=[("what was going on with the cat", [1, 2], "trajectory")],
    ),
    dict(
        id="boiler", fam="home",
        eps=[
            "No hot water this morning and the boiler is showing an error code.",
            "The engineer said the pressure had dropped and topped it up.",
            "He warned the expansion vessel will need replacing within a year.",
        ],
        qs=[("what did the engineer say I would need later", [2], "caution")],
    ),
    dict(
        id="mother", fam="family",
        eps=[
            "Mum mentioned her hearing has got worse on the phone.",
            "Booked her an appointment at the audiology clinic for the ninth.",
            "The test showed moderate loss and they are fitting an aid.",
            "She is finding the aid uncomfortable and keeps taking it out.",
        ],
        qs=[("how is my mother getting on", [2, 3], "trajectory"),
            ("what did I arrange for her", [1], "commitment")],
    ),
    dict(
        id="wedding", fam="family",
        eps=[
            "Sam's wedding is in June and I am expected to speak.",
            "The venue is two hours away so an overnight stay is needed.",
            "Drafted the speech and it is far too long at eleven minutes.",
        ],
        qs=[("what do I still need to do for the wedding", [0, 2], "todo")],
    ),
    dict(
        id="visa", fam="admin",
        eps=[
            "The visa application needs a bank statement covering six months.",
            "Uploaded everything except the employment letter, which HR still owes me.",
            "HR sent the letter but dated it wrongly and it was rejected.",
            "Resubmitted with a corrected letter and the application went through.",
        ],
        qs=[("what held up the visa", [2], "fact"),
            ("how did the paperwork end up", [1, 2, 3], "trajectory")],
    ),
    dict(
        id="passport", fam="admin",
        eps=[
            "The passport expires in eight months, which is inside the six-month rule for some countries.",
            "Renewal is running at five weeks according to the site.",
        ],
        qs=[("is there a problem with my travel documents", [0, 1], "caution")],
    ),
    dict(
        id="sleep", fam="health",
        eps=[
            "Waking at four most nights and not getting back off.",
            "The GP suggested cutting caffeine after midday before trying anything else.",
            "Two weeks without afternoon coffee and the waking has mostly stopped.",
        ],
        qs=[("what did I try for the sleeping problem", [1, 2], "trajectory")],
    ),
    dict(
        id="allergy", fam="health",
        eps=[
            "Came out in a rash after the seafood at the restaurant.",
            "The doctor thinks shellfish rather than the sauce.",
            "Told to carry antihistamines and avoid it entirely until tested.",
        ],
        qs=[("what was I told to do about the reaction", [2], "instruction")],
    ),
    dict(
        id="promotion", fam="work",
        eps=[
            "Was told the promotion round opens in September.",
            "Need three examples of independent project work for the case.",
            "The manager offered to write a supporting statement.",
            "Submitted the case a day before the deadline.",
        ],
        qs=[("what did I need to put together for the promotion", [1, 2], "todo")],
    ),
    dict(
        id="conference", fam="work",
        eps=[
            "The abstract deadline is the eighteenth and the paper a week after.",
            "Registration is cheaper before the end of the month.",
            "The talk slot is twenty minutes including questions.",
        ],
        qs=[("what are the dates I need to hit", [0, 1], "fact")],
    ),
    dict(
        id="bike", fam="hobby",
        eps=[
            "The rear derailleur is skipping under load on the climbs.",
            "The shop says the hanger is bent, probably from the fall.",
            "Straightened and reindexed and the shifting is clean again.",
        ],
        qs=[("what was wrong with the bicycle", [1], "fact")],
    ),
    dict(
        id="spanish", fam="hobby",
        eps=[
            "Started the Spanish course; the subjunctive is where everyone stalls.",
            "The tutor recommends reading rather than more grammar drills.",
            "Got through a short novel and the grammar started making sense in context.",
        ],
        qs=[("what helped with the language learning", [1, 2], "trajectory")],
    ),
    dict(
        id="pension", fam="finance",
        eps=[
            "The pension statement shows the default fund has underperformed.",
            "Switching funds is free once a year and I have not used it.",
            "Moved to the lower-fee tracker after reading the comparison.",
        ],
        qs=[("what did I change about the pension", [1, 2], "decision")],
    ),
    dict(
        id="mortgage", fam="finance",
        eps=[
            "The fixed rate ends in fourteen months and rates have moved up.",
            "The broker says we can lock in six months ahead without a fee.",
            "Booked a rate to hold it, with the option to drop it if things improve.",
        ],
        qs=[("what is the plan on the mortgage", [1, 2], "decision")],
    ),
    dict(id="roof", fam="home", eps=[
        "Water stain appeared on the ceiling in the back bedroom after the storm.",
        "The roofer found two slipped tiles above that corner.",
        "Quoted four hundred to replace them and reseal the flashing.",
        "Work done on Tuesday and the stain has not spread since."],
        qs=[("what caused the damp patch upstairs",[1],"fact"),
            ("how did the ceiling problem get sorted",[2,3],"trajectory")]),
    dict(id="garden", fam="home", eps=[
        "The apple tree has not fruited for two years running.",
        "Neighbour thinks it needs a pollination partner nearby.",
        "Planted a crab apple at the end of the bed in October.",
        "First proper crop this autumn, so the theory held."],
        qs=[("why was the tree not producing",[1],"fact"),
            ("what did I do about the fruit tree",[2,3],"trajectory")]),
    dict(id="damp", fam="home", eps=[
        "Black mould keeps returning in the corner of the bathroom.",
        "The surveyor said it is condensation, not rising damp.",
        "Fitted an extractor with a humidity sensor and it has stayed clear."],
        qs=[("what was the actual cause of the mould",[1],"fact")]),
    dict(id="locks", fam="home", eps=[
        "Lost the spare key somewhere between the office and the station.",
        "Locksmith advised changing the barrel rather than just cutting a new key.",
        "New barrel fitted and three keys cut."],
        qs=[("what did the locksmith recommend",[1],"instruction")]),
    dict(id="dishwasher", fam="home", eps=[
        "The dishwasher is leaving grit on the glasses.",
        "Cleaned the filter, which was completely blocked.",
        "Better but not perfect, so switched to a different salt."],
        qs=[("what did I try for the dishwasher",[1,2],"trajectory")]),
    dict(id="tax", fam="finance", eps=[
        "The self-assessment deadline is the end of January.",
        "Accountant needs the receipts by the first week of December.",
        "Missing the receipts from the March trip entirely.",
        "Reconstructed them from the card statement and it was accepted."],
        qs=[("what did I owe the accountant and when",[1],"commitment"),
            ("how did the missing paperwork resolve",[2,3],"trajectory")]),
    dict(id="isa", fam="finance", eps=[
        "The cash ISA rate dropped to almost nothing after the bonus year ended.",
        "Compared transfers and one provider pays nearly double.",
        "Started the transfer, which they said takes fifteen working days."],
        qs=[("what am I doing about the savings account",[1,2],"decision")]),
    dict(id="salary", fam="finance", eps=[
        "Pay review happens in April and the pot is smaller this year.",
        "Was told my case is strong but the timing is bad.",
        "Got a smaller rise than expected with a promise to revisit in six months."],
        qs=[("how did the pay conversation go",[1,2],"trajectory")]),
    dict(id="creditcard", fam="finance", eps=[
        "Noticed a subscription I cancelled is still being charged.",
        "The company claims the cancellation never went through.",
        "The bank did a chargeback and refunded three months."],
        qs=[("what happened with the wrong charges",[1,2],"trajectory")]),
    dict(id="dissertation", fam="work", eps=[
        "The literature review is the weakest chapter by some distance.",
        "Advised to organise it by argument rather than chronologically.",
        "Restructured around three competing positions and it works far better.",
        "Word count is now over by two thousand."],
        qs=[("what was wrong with the review chapter",[0,1],"feedback"),
            ("what problem do I still have",[3],"todo")]),
    dict(id="teaching", fam="work", eps=[
        "Assigned two tutorial groups this term instead of one.",
        "The Thursday group is much quieter and harder to get talking.",
        "Tried small-group work first and the discussion improved a lot."],
        qs=[("how did I handle the difficult tutorial",[1,2],"trajectory")]),
    dict(id="review_paper", fam="work", eps=[
        "Asked to review a paper for the workshop by the tenth.",
        "The paper has a serious flaw in the evaluation setup.",
        "Recommended major revisions rather than rejection."],
        qs=[("what did I decide about the paper I reviewed",[1,2],"decision")]),
    dict(id="collab", fam="work", eps=[
        "The collaboration with the Munich group is stalling on data access.",
        "Their legal team needs a data sharing agreement first.",
        "Agreement signed after six weeks and the data arrived."],
        qs=[("what was holding up the joint project",[1],"fact"),
            ("did the collaboration get moving",[1,2],"trajectory")]),
    dict(id="interview", fam="work", eps=[
        "Have an interview for the research post on the twenty-eighth.",
        "They want a twenty-minute presentation on prior work.",
        "Practised it twice and it runs to twenty-six minutes."],
        qs=[("what do I need to prepare for the interview",[1,2],"todo")]),
    dict(id="conference_travel", fam="travel", eps=[
        "Flights to the conference are only refundable at the higher fare.",
        "Booked the flexible fare because the dates might move.",
        "Dates did move by a day and the change cost nothing."],
        qs=[("was the expensive ticket worth it",[1,2],"trajectory")]),
    dict(id="iceland", fam="travel", eps=[
        "Considering Iceland in February for the northern lights.",
        "Read that the ring road is often closed that time of year.",
        "Decided to base in one place rather than drive around."],
        qs=[("how did the Iceland plan change",[1,2],"revision")]),
    dict(id="hotel", fam="travel", eps=[
        "The hotel charged for a night I had cancelled within the window.",
        "Emailed twice with the confirmation and got no reply.",
        "Disputed it with the card company and it was reversed."],
        qs=[("how did I get the hotel money back",[2],"fact")]),
    dict(id="trainstrike", fam="travel", eps=[
        "Strike announced for the day I was meant to travel down.",
        "Rebooked onto the coach which takes twice as long.",
        "Strike was called off two days before but the coach was non-refundable."],
        qs=[("what went wrong with the travel plans",[1,2],"trajectory")]),
    dict(id="physio2", fam="health", eps=[
        "Shoulder has been clicking since the decorating.",
        "Physio says it is impingement and not a tear.",
        "Given three exercises and told to avoid overhead lifting.",
        "Much better after a month but still clicks occasionally."],
        qs=[("what was diagnosed with the shoulder",[1],"fact"),
            ("what was I told to avoid",[2],"instruction")]),
    dict(id="bloodtest", fam="health", eps=[
        "Routine bloods came back with low iron.",
        "GP suggested supplements and a recheck in three months.",
        "Recheck showed levels back in range."],
        qs=[("what came out of the blood test",[0,1],"fact")]),
    dict(id="migraine", fam="health", eps=[
        "Headaches have been more frequent since the schedule changed.",
        "Keeping a diary showed they cluster after poor sleep.",
        "Fixed the sleep pattern and the frequency dropped by half."],
        qs=[("what triggered the headaches",[1],"fact"),
            ("what helped with the headaches",[1,2],"trajectory")]),
    dict(id="eyes", fam="health", eps=[
        "Screen work is leaving my eyes tired by the afternoon.",
        "Optician found a slight prescription change and suggested a screen coating.",
        "New glasses have made a noticeable difference."],
        qs=[("what did the optician suggest",[1],"instruction")]),
    dict(id="vaccination", fam="health", eps=[
        "Due the booster before travelling in November.",
        "The clinic only does them on Wednesday mornings.",
        "Had it done and the arm was sore for two days."],
        qs=[("what did I need to arrange before the trip",[0,1],"todo")]),
    dict(id="piano", fam="hobby", eps=[
        "Restarted piano after about ten years away.",
        "Sight reading has gone completely but the muscle memory is there.",
        "Teacher suggested working through easy pieces rather than hard ones."],
        qs=[("what advice did the piano teacher give",[2],"instruction")]),
    dict(id="chess", fam="hobby", eps=[
        "Rating has been stuck around the same level for months.",
        "Analysis shows most losses come from the same opening.",
        "Switched openings and the results improved within a few weeks."],
        qs=[("why was I losing games",[1],"fact"),
            ("what did I change about chess",[1,2],"trajectory")]),
    dict(id="pottery", fam="hobby", eps=[
        "Signed up for the evening pottery course starting in September.",
        "Centring the clay is much harder than it looks.",
        "By week six the bowls are finally symmetrical."],
        qs=[("how is the pottery going",[1,2],"trajectory")]),
    dict(id="photography", fam="hobby", eps=[
        "The lens has developed a fungus spot in storage.",
        "Repair costs almost as much as a used replacement.",
        "Bought a used copy and kept the old one for parts."],
        qs=[("what did I do about the camera lens",[1,2],"decision")]),
    dict(id="swimming", fam="hobby", eps=[
        "Joined the pool to give the knee a break from running.",
        "Front crawl breathing is the limiting factor, not fitness.",
        "A few technique sessions and I can do a kilometre comfortably."],
        qs=[("what was limiting my swimming",[1],"fact")]),
    dict(id="phone", fam="tech", eps=[
        "The phone battery is now dying by mid-afternoon.",
        "Replacing the battery is a third of the price of a new phone.",
        "Had the battery swapped and it lasts all day again."],
        qs=[("what did I do about the phone",[1,2],"decision")]),
    dict(id="wifi", fam="tech", eps=[
        "The connection drops in the back room every evening.",
        "Engineer found the router channel was clashing with the neighbours.",
        "Changed the channel and added a mesh point, and it is stable now."],
        qs=[("why was the internet dropping",[1],"fact"),
            ("how did the connection get fixed",[1,2],"trajectory")]),
    dict(id="email", fam="tech", eps=[
        "The mailbox is full and rejecting incoming messages.",
        "Most of the space is old attachments from the project.",
        "Archived them to the drive and freed most of the quota."],
        qs=[("what filled up the mailbox",[1],"fact")]),
    dict(id="monitor", fam="tech", eps=[
        "The second monitor flickers when the machine wakes from sleep.",
        "Swapping the cable made no difference.",
        "A firmware update on the monitor fixed it."],
        qs=[("what actually fixed the display problem",[1,2],"trajectory")]),
    dict(id="cloud", fam="tech", eps=[
        "The storage plan renews next month at a higher price.",
        "Most of what is stored is duplicated from the laptop anyway.",
        "Downgraded to the smaller tier after clearing duplicates."],
        qs=[("what did I change about the storage",[1,2],"decision")]),
    dict(id="brother", fam="family", eps=[
        "Tom is thinking about moving back closer to home.",
        "His job would let him work remotely three days a week.",
        "He has put an offer in on a place twenty minutes away."],
        qs=[("what is my brother planning",[0,2],"trajectory")]),
    dict(id="childcare", fam="family", eps=[
        "The nursery is putting fees up from April.",
        "Looked at two others but both have long waiting lists.",
        "Staying put and adjusting the budget instead."],
        qs=[("what did we decide about nursery",[1,2],"decision")]),
    dict(id="anniversary", fam="family", eps=[
        "Parents' fortieth is in September and nothing is booked.",
        "They want something small rather than a big party.",
        "Booked a lunch for twelve at the place by the river."],
        qs=[("what are we doing for the anniversary",[1,2],"decision")]),
    dict(id="grandmother", fam="family", eps=[
        "Gran had a fall in the kitchen but nothing was broken.",
        "The GP arranged an occupational therapy assessment.",
        "They fitted grab rails and she is much steadier."],
        qs=[("what happened to my grandmother",[0,1],"fact"),
            ("what was put in place for her",[1,2],"trajectory")]),
    dict(id="school", fam="family", eps=[
        "Parents evening flagged that maths is falling behind.",
        "Teacher suggested a weekly tutor rather than more homework.",
        "Six weeks of tutoring and the last test was much better."],
        qs=[("what was suggested about the schoolwork",[1],"instruction")]),
    dict(id="petinsurance", fam="admin", eps=[
        "Pet insurance renewal excludes anything already claimed for.",
        "The vet bills last year were mostly the one condition.",
        "Switched to a lifetime policy despite the higher premium."],
        qs=[("why did I change pet insurance",[0,2],"decision")]),
    dict(id="driving", fam="admin", eps=[
        "The licence photo expires in three months.",
        "Renewal can be done online if the passport photo is recent.",
        "Renewed online and the new one arrived in a week."],
        qs=[("how did I renew the licence",[1,2],"trajectory")]),
    dict(id="council", fam="admin", eps=[
        "The council sent a bill for a property I moved out of.",
        "Sent the tenancy end date and they said it would be corrected.",
        "A reminder arrived anyway so I had to call again.",
        "Finally cleared after the third contact."],
        qs=[("what went wrong with the council bill",[0,2],"trajectory"),
            ("how many times did I have to chase it",[2,3],"fact")]),
    dict(id="broadband", fam="admin", eps=[
        "Broadband contract ends in two months and the price doubles after.",
        "A competitor offers the same speed for less with a router included.",
        "Switched, and the changeover had one day of downtime."],
        qs=[("what did I do about the broadband contract",[1,2],"decision")]),
    dict(id="parking", fam="admin", eps=[
        "Got a parking ticket outside the clinic despite the permit.",
        "The permit had slipped off the dashboard onto the floor.",
        "Appealed with a photo of the permit and it was cancelled."],
        qs=[("how did I deal with the parking fine",[1,2],"trajectory")]),
    dict(id="restaurant", fam="food", eps=[
        "The tasting menu place needs booking two months ahead.",
        "Got a cancellation slot for a Tuesday in October.",
        "It was good but not worth the price honestly."],
        qs=[("what did I think of the restaurant",[2],"opinion")]),
    dict(id="diet", fam="food", eps=[
        "Cutting back on afternoon sugar to stop the energy crash.",
        "Swapped to fruit and nuts and the crash is much smaller.",
        "Weekends are still the difficult part."],
        qs=[("what change did I make to eating",[0,1],"trajectory")]),
    dict(id="wine", fam="food", eps=[
        "The case from the merchant arrived with two broken bottles.",
        "They replaced them without asking for the damaged ones back."],
        qs=[("what happened with the wine order",[0,1],"trajectory")]),
    dict(id="allotment", fam="food", eps=[
        "Got to the top of the allotment waiting list finally.",
        "The plot is completely overgrown with bindweed.",
        "Cleared half of it and covered the rest to kill it off."],
        qs=[("what state was the allotment in",[1],"fact"),
            ("what have I done with the plot so far",[2],"trajectory")]),
    dict(id="cooking", fam="food", eps=[
        "The oven runs about twenty degrees cold according to the thermometer.",
        "Adjusting the dial up has fixed the underbaking."],
        qs=[("what was wrong with the baking",[0,1],"trajectory")]),

    dict(id="movehouse", fam="property", eps=[
        "The survey on the new place flagged old wiring throughout.",
        "Electrician quoted for a partial rewire of the ground floor.",
        "Negotiated half the cost off the asking price.",
        "Rewire booked for the week after completion."],
        qs=[("what did the survey turn up",[0],"fact"),
            ("how did the wiring issue get resolved",[2,3],"trajectory")]),
    dict(id="tenant", fam="property", eps=[
        "The tenants gave notice a month earlier than expected.",
        "Agent suggested redecorating between lets to raise the rent.",
        "Painted throughout and relet within two weeks."],
        qs=[("what did the letting agent advise",[1],"instruction")]),
    dict(id="neighbour_dispute", fam="property", eps=[
        "The fence between us has been leaning since the winter.",
        "Deeds show it is our responsibility, not theirs.",
        "Split the cost anyway to keep things civil."],
        qs=[("whose fence was it legally",[1],"fact"),
            ("how did the fence get handled",[1,2],"trajectory")]),
    dict(id="solar", fam="property", eps=[
        "Getting quotes for solar panels on the south roof.",
        "Payback period works out around nine years at current rates.",
        "Decided to wait until the roof is redone anyway."],
        qs=[("why did I hold off on solar",[1,2],"decision")]),
    dict(id="thesis_viva", fam="work", eps=[
        "Viva scheduled for the second week of March.",
        "Examiners are one internal and one from Edinburgh.",
        "Told to expect questions on the evaluation design above all.",
        "Passed with minor corrections due in three months."],
        qs=[("what was I warned to prepare for",[2],"caution"),
            ("how did the viva go",[3],"fact")]),
    dict(id="funding_report", fam="work", eps=[
        "The funder wants an interim report by the end of the quarter.",
        "It needs financials that finance have not produced yet.",
        "Got an extension of two weeks after explaining the delay."],
        qs=[("what held up the funder report",[1],"fact")]),
    dict(id="hiring", fam="work", eps=[
        "We are recruiting a research assistant for the project.",
        "Only three of the eleven applicants had the stats background.",
        "Offered to the second choice after the first declined."],
        qs=[("how did the recruitment go",[1,2],"trajectory")]),
    dict(id="ethics", fam="work", eps=[
        "The ethics application was returned for more detail on consent.",
        "Rewrote the consent form in plain language.",
        "Approved on the second submission."],
        qs=[("what did the ethics committee want changed",[0,1],"feedback")]),
    dict(id="codebase", fam="work", eps=[
        "The analysis scripts have grown unmanageable across three folders.",
        "Spent a day consolidating into one pipeline with a config file.",
        "Rerunning everything from scratch now takes twenty minutes."],
        qs=[("what did I do about the messy code",[1,2],"trajectory")]),
    dict(id="conference_talk", fam="work", eps=[
        "The talk was accepted but moved to the last slot of the day.",
        "Half the room had left by then.",
        "Two people followed up by email afterwards anyway."],
        qs=[("how did the presentation go",[1,2],"trajectory")]),
    dict(id="backpain", fam="health", eps=[
        "Lower back started aching after the long drive.",
        "GP said it is muscular and to keep moving rather than rest.",
        "Two weeks of walking and it has settled."],
        qs=[("what was I told to do about my back",[1],"instruction")]),
    dict(id="flu", fam="health", eps=[
        "Came down with something the week of the deadline.",
        "Lost four working days entirely.",
        "Asked for and got a short extension."],
        qs=[("what did being ill cost me",[1,2],"trajectory")]),
    dict(id="skin", fam="health", eps=[
        "The rash on my hands is worse in winter.",
        "Dermatologist says it is contact dermatitis from the cleaning products.",
        "Switched to gloves and a different detergent and it cleared."],
        qs=[("what was causing the hand problem",[1],"fact")]),
    dict(id="cholesterol", fam="health", eps=[
        "Cholesterol came back slightly raised at the check-up.",
        "Advised diet changes before considering medication.",
        "Retest in six months showed it back down."],
        qs=[("what was the plan for the cholesterol",[1,2],"trajectory")]),
    dict(id="counselling", fam="health", eps=[
        "Waiting list for the service is around four months.",
        "Went private for the first few sessions instead.",
        "The NHS slot came through sooner than quoted."],
        qs=[("how did I get seen",[1,2],"trajectory")]),
    dict(id="norway", fam="travel", eps=[
        "Looking at the fjords in summer rather than winter.",
        "Ferries need booking well ahead in July.",
        "Shifted to late August when it is quieter and cheaper."],
        qs=[("when did I settle on going",[2],"decision")]),
    dict(id="luggage", fam="travel", eps=[
        "The airline lost my bag on the connection.",
        "It turned up two days later at the hotel.",
        "Claimed for the clothes I had to buy and got most of it back."],
        qs=[("what happened with the lost luggage",[1,2],"trajectory")]),
    dict(id="roadtrip", fam="travel", eps=[
        "Planning to drive down through France in the spring.",
        "The car needs a service before a trip that long.",
        "Booked the service and bought the breakdown cover."],
        qs=[("what did I need to sort before driving",[1,2],"todo")]),
    dict(id="visa_esta", fam="travel", eps=[
        "The travel authorisation for the US expired without me noticing.",
        "Reapplied and it came through in under an hour."],
        qs=[("what nearly went wrong with the US trip",[0],"caution")]),
    dict(id="cottage", fam="travel", eps=[
        "Booked the cottage for the week after Easter.",
        "It turns out to be a forty minute drive from any shop.",
        "Took most of the food with us and it was fine."],
        qs=[("what was the problem with the cottage",[1],"fact")]),
    dict(id="knitting", fam="hobby", eps=[
        "Started the cardigan and the tension is off on the sleeves.",
        "Went down a needle size and reknit the second sleeve.",
        "They match now but the first sleeve has to be redone."],
        qs=[("what went wrong with the knitting",[0,2],"trajectory")]),
    dict(id="woodwork", fam="hobby", eps=[
        "The shelf brackets are not sitting flush against the wall.",
        "The wall itself is out of true, not the brackets.",
        "Packed them out with shims and it looks fine."],
        qs=[("why were the shelves not level",[1],"fact")]),
    dict(id="cycling_club", fam="hobby", eps=[
        "Joined the Sunday club run for the first time.",
        "Got dropped on the second climb and rode back alone.",
        "Six weeks later I finished with the group."],
        qs=[("how did the club riding progress",[1,2],"trajectory")]),
    dict(id="birdwatching", fam="hobby", eps=[
        "The feeder has attracted mostly starlings and little else.",
        "Changed to sunflower hearts and the variety improved."],
        qs=[("what change brought more birds",[1],"fact")]),
    dict(id="film_club", fam="hobby", eps=[
        "The film society meets fortnightly on Wednesdays.",
        "Volunteered to pick the programme for the spring term.",
        "Chose a run of six that nobody had seen before."],
        qs=[("what did I take on at the film society",[1,2],"commitment")]),
    dict(id="laptop_work", fam="tech", eps=[
        "IT are replacing all the machines with locked-down builds.",
        "The new build blocks the tools I need for the analysis.",
        "Got an exception approved after escalating twice."],
        qs=[("what problem did the new machines cause",[1],"fact"),
            ("how did the IT issue get resolved",[2],"trajectory")]),
    dict(id="printer", fam="tech", eps=[
        "The printer refuses third party cartridges after the update.",
        "Rolled the firmware back and it accepts them again."],
        qs=[("how did I fix the printer",[1],"fact")]),
    dict(id="passwords", fam="tech", eps=[
        "The old password manager is discontinuing the free tier.",
        "Exported everything and imported into a different one.",
        "A handful of entries lost their notes in the transfer."],
        qs=[("what went wrong in the migration",[2],"fact")]),
    dict(id="smarthome", fam="tech", eps=[
        "The heating app stopped talking to the thermostat after the update.",
        "Repairing it required resetting the whole hub.",
        "Working again but all the schedules had to be re-entered."],
        qs=[("what did fixing the heating app cost me",[1,2],"trajectory")]),
    dict(id="phone_plan", fam="tech", eps=[
        "Contract is out of its minimum term and the price has not dropped.",
        "A SIM only deal is a third of what I pay now.",
        "Switched and kept the number."],
        qs=[("what did I change about the phone bill",[1,2],"decision")]),
    dict(id="uncle", fam="family", eps=[
        "Uncle Ray has been in hospital with a chest infection.",
        "He is home now but needs help with shopping for a while.",
        "We have set up a rota between the three of us."],
        qs=[("what support was arranged",[1,2],"trajectory")]),
    dict(id="cousin_wedding", fam="family", eps=[
        "The wedding abroad clashes with the conference week.",
        "Managed to move the talk to a colleague.",
        "Going to both, just with a very tight turnaround."],
        qs=[("how did I resolve the clash",[1,2],"trajectory")]),
    dict(id="inlaws", fam="family", eps=[
        "They are visiting for a fortnight in the summer.",
        "The spare room needs a proper bed rather than the sofa bed.",
        "Bought one in the sale and it arrives before they do."],
        qs=[("what did I need to buy before the visit",[1,2],"todo")]),
    dict(id="sibling_money", fam="family", eps=[
        "Lent my sister money for the deposit last year.",
        "She has been paying it back monthly without being asked.",
        "Two payments left as of this month."],
        qs=[("how is the loan going",[1,2],"trajectory")]),
    dict(id="dog", fam="family", eps=[
        "The dog has started barking whenever we leave the house.",
        "Trainer says it is separation anxiety and to build up slowly.",
        "Started with two minutes at a time and it is improving."],
        qs=[("what did the trainer diagnose",[1],"fact"),
            ("what approach are we taking with the dog",[1,2],"instruction")]),
    dict(id="gas", fam="admin", eps=[
        "The energy supplier estimated the bill far too high again.",
        "Submitted actual readings and got a large credit back."],
        qs=[("what was wrong with the energy bill",[0,1],"trajectory")]),
    dict(id="prescription", fam="admin", eps=[
        "The repeat prescription keeps being sent to the old pharmacy.",
        "Updated it at the surgery in person because the app would not.",
        "Next one went to the right place."],
        qs=[("how did I fix the prescription problem",[1,2],"trajectory")]),
    dict(id="post", fam="admin", eps=[
        "Mail for the previous occupant is still arriving daily.",
        "Returned a batch marked not known at this address.",
        "It has mostly stopped after about two months."],
        qs=[("what did I do about the wrong post",[1,2],"trajectory")]),
    dict(id="warranty", fam="admin", eps=[
        "The washing machine failed five weeks out of warranty.",
        "Manufacturer offered a goodwill repair after I pushed.",
        "Repaired free but I had to wait three weeks for the part."],
        qs=[("how did the warranty situation end",[1,2],"trajectory")]),
    dict(id="subscription", fam="admin", eps=[
        "Counted six subscriptions I had forgotten about.",
        "Cancelled four and kept the two I actually use."],
        qs=[("what did I do about the subscriptions",[1],"decision")]),
    dict(id="bakery", fam="food", eps=[
        "The bakery on the high street has closed down.",
        "The replacement is a chain and the bread is much worse.",
        "Started making it at home more often as a result."],
        qs=[("what changed about where I get bread",[0,2],"trajectory")]),
    dict(id="veg_box", fam="food", eps=[
        "Signed up for the weekly vegetable box.",
        "Getting far more than two of us can eat.",
        "Switched to fortnightly and it works much better."],
        qs=[("what was the problem with the veg box",[1],"fact")]),
    dict(id="cheese", fam="food", eps=[
        "The cheese order arrived warm because of the delivery delay.",
        "They refunded it in full without a fuss."],
        qs=[("what happened with the cheese delivery",[0,1],"trajectory")]),
    dict(id="lunch", fam="food", eps=[
        "Buying lunch every day was costing more than I realised.",
        "Started batch cooking on Sundays instead.",
        "Saving roughly forty a week and eating better."],
        qs=[("what change did I make about lunches",[1,2],"trajectory")]),

]

# Filler memories -- realistic but not part of any question's gold set. They exist so the
# corpus has the density of a real store and so retrieval has something to get wrong.
FILLER = [
    "Picked up the dry cleaning on the way home.",
    "The delivery was left with the neighbour at number twelve.",
    "Rain most of the afternoon so the walk was cut short.",
    "Called the bank about the duplicate charge and it was refunded.",
    "The library book is due back on the twenty-third.",
    "Tried the new place on the corner; the service was slow but the food was good.",
    "Meeting moved to Thursday morning at the last minute.",
    "The printer is out of cyan again.",
    "Watched the second half of the match at Dan's.",
    "Bins go out on Tuesday night now, not Wednesday.",
    "The bus route has changed and no longer stops at the bridge.",
    "Bought tickets for the exhibition before it sells out.",
    "The plant on the windowsill is finally putting out new leaves.",
    "Forgot to reply to the message about the weekend.",
    "The pharmacy had the prescription ready early for once.",
    "Chased the invoice that has been outstanding since last month.",
    "The heating comes on too early in the morning; adjusted the timer.",
    "Ran into Priya at the station and we agreed to get coffee.",
    "The parcel tracking says it is still at the depot.",
    "Cleaned out the shed and took two loads to the tip.",
    "The fridge is making a clicking noise at night.",
    "Signed the birthday card going round the office.",
    "Recycling collection was missed this week.",
    "The password manager flagged three reused logins.",
    "Booked a haircut for Saturday morning.",
    "The neighbour's scaffolding is finally coming down.",
    "Sent the photos from the trip to the family group.",
    "The seedlings need potting on before they get leggy.",
    "Paid the council tax instalment.",
    "The train was delayed twenty minutes because of signalling.",
]

# Paraphrase noise: near-duplicates of gold episodes that are NOT gold. These are the
# hardest distractors -- semantically adjacent, factually different.
NEAR_MISS = [
    ("apartment", "A friend signed a lease on a two-bedroom near Warren Street last year."),
    ("knee", "A colleague was told to rest his shoulder for six weeks after an injury."),
    ("thesis", "Another student was told to expand her introduction rather than cut it."),
    ("japan", "Considered Osaka last year but went to Seoul instead."),
    ("car", "The other car needed a new sensor two winters ago."),
    ("sourdough", "The rye starter behaves completely differently in the cold."),
    ("mother", "Dad had a hearing test years ago and did not need an aid."),
    ("laptop", "The work laptop battery is still fine after three years."),
    ("boiler", "The old flat had a boiler that lost pressure constantly."),
    ("backup", "The office backup system failed during the migration last spring."),
]



# ---- compositional filler ------------------------------------------------------------
# Realistic but question-irrelevant. A real memory store is mostly this, and retrieval
# has to reject it. Hand-writing 1000 sentences is not a good use of anyone's time;
# these compose from natural fragments and read like ordinary log entries.
_SUBJ = ["The parcel", "The report", "The meeting", "The invoice", "The appointment",
         "The delivery", "The form", "The refund", "The booking", "The renewal",
         "The paperwork", "The order", "The inspection", "The transfer", "The quote"]
_VERB = ["was rescheduled to", "has been delayed until", "was confirmed for",
         "needs to be finished by", "was moved forward to", "is outstanding since",
         "was cancelled on", "arrives on", "was chased again on", "is due on"]
_WHEN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "the weekend",
         "the fifth", "the eleventh", "the nineteenth", "the twenty-second",
         "next week", "the end of the month", "the following Monday", "the third"]
_TAIL = ["", " and nobody told the team", " which suits me better", " after two reminders",
         " because of the holiday", " so the plan has shifted", " with no explanation",
         " though it may slip again", " and it is now in the calendar",
         " which leaves very little time"]
_MISC = [
    "Weather was {} all day so the plans changed.",
    "Spoke to {} about the arrangements and nothing is settled.",
    "The {} needs replacing before winter.",
    "Left a message for {} and heard nothing back.",
    "Ordered {} online because the shop had none.",
    "The {} is louder than it used to be.",
]
_MISCFILL = ["grim", "unexpectedly warm", "Rob", "the landlord", "the letting agent",
             "gutter", "back door", "extractor fan", "Anna", "the plumber",
             "replacement filters", "printer paper", "a new kettle", "washing machine",
             "radiator", "shed roof", "front gate", "the surveyor"]


def _compositional(rng, n):
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < n * 40:
        guard += 1
        if rng.random() < 0.72:
            t = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_WHEN)}{rng.choice(_TAIL)}."
        else:
            t = rng.choice(_MISC).format(rng.choice(_MISCFILL))
        if t not in seen:
            seen.add(t); out.append(t)
    return out


def build(seed: int = 0, n_filler: int | None = None):
    """Return (episodes, questions). Deterministic given the seed."""
    rng = random.Random(seed)
    eps: list[Episode] = []
    qs: list[Question] = []

    threads = list(THREADS)
    rng.shuffle(threads)

    # interleave threads in time -- a real store is not grouped by topic
    pending = [(th, list(enumerate(th["eps"]))) for th in threads]
    t = 0
    index_of: dict[tuple[str, int], int] = {}
    while any(p[1] for p in pending):
        rng.shuffle(pending)
        for th, items in pending:
            if not items:
                continue
            local_i, text = items.pop(0)
            index_of[(th["id"], local_i)] = len(eps)
            eps.append(Episode(text=text, thread=th["id"], t=t))
            t += 1

    for th in threads:
        for qtext, local_ids, kind in th["qs"]:
            gold = [index_of[(th["id"], i)] for i in local_ids]
            qs.append(Question(text=qtext, gold_idx=gold, thread=th["id"], kind=kind))

    for thread_id, text in NEAR_MISS:
        eps.append(Episode(text=text, thread=thread_id + "_nearmiss", t=t)); t += 1

    for text in FILLER:
        eps.append(Episode(text=text, thread="filler", t=t)); t += 1
    if n_filler:
        for text in _compositional(rng, n_filler):
            eps.append(Episode(text=text, thread="filler", t=t)); t += 1

    return eps, qs
