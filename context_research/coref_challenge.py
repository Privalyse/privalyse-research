"""
Coreference Challenge Set

A targeted dataset where generic tags (<PERSON>) fail but semantic surrogates succeed.

These test cases require:
1. Multiple people in one text
2. Referential integrity questions
3. Cross-document reasoning

This is the "smoking gun" benchmark that demonstrates semantic masking's advantage.
"""

import sys
import os
import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CorefTestCase:
    """A single coreference test case."""
    id: str
    text: str
    question: str
    answer: str
    requires_entity_distinction: bool
    difficulty: str  # "easy", "medium", "hard"
    category: str  # "team", "family", "conversation", "multi_doc"


def generate_team_scenarios(rng: random.Random) -> List[CorefTestCase]:
    """Generate scenarios with multiple team members."""
    
    cases = []
    
    # Team leadership scenario
    cases.append(CorefTestCase(
        id="team_001",
        text="""
Project Update - Q4 Planning

Alice Chen is the team lead for the infrastructure project. Bob Martinez handles the frontend work, 
while Carol Williams manages backend services. In yesterday's standup, Alice assigned the API 
refactoring to Bob. Carol will review Bob's changes before the Friday deadline.

Alice mentioned that Bob's previous work on the authentication module was excellent, which is 
why she's entrusting him with this critical task. Carol agreed, noting that Bob's code quality 
has been consistently high.

Action items:
- Bob: Complete API refactoring by Thursday
- Carol: Code review by Friday morning
- Alice: Final sign-off by Friday EOD
""".strip(),
        question="Who is responsible for the code review?",
        answer="Carol Williams",
        requires_entity_distinction=True,
        difficulty="easy",
        category="team"
    ))
    
    cases.append(CorefTestCase(
        id="team_002",
        text="""
Meeting Notes - Design Review

Participants: David Kim (Design Lead), Emma Roberts (UX), Frank Wilson (Product)

David presented the new dashboard mockups. Emma raised concerns about the color contrast, 
suggesting we use darker tones for accessibility. Frank agreed with Emma's feedback and 
asked David to incorporate the changes.

David will update the designs and share with Emma for a second review. Frank requested 
that Emma sign off before we move to development.

Decision: Emma has final approval authority on UX changes.
""".strip(),
        question="Who has final approval authority on UX changes?",
        answer="Emma Roberts",
        requires_entity_distinction=True,
        difficulty="easy",
        category="team"
    ))
    
    cases.append(CorefTestCase(
        id="team_003",
        text="""
Incident Report - Database Outage

At 3:42 PM, Grace Liu (on-call engineer) detected the outage. She immediately paged 
Henry Park (DBA) and Ivan Chen (Platform Lead).

Henry identified the root cause: a misconfigured connection pool. Ivan asked Grace to 
implement the hotfix since she had written the original connection handling code.

Grace deployed the fix at 4:15 PM. Henry verified the database was healthy. Ivan 
conducted the post-mortem and credited Grace for the quick resolution.

Commendation: Grace Liu for rapid incident response.
""".strip(),
        question="Who deployed the hotfix?",
        answer="Grace Liu",
        requires_entity_distinction=True,
        difficulty="medium",
        category="team"
    ))
    
    return cases


def generate_family_scenarios(rng: random.Random) -> List[CorefTestCase]:
    """Generate family relationship scenarios."""
    
    cases = []
    
    cases.append(CorefTestCase(
        id="family_001",
        text="""
Family Gathering - Holiday Plans

The Johnsons are planning their annual Thanksgiving dinner. Margaret Johnson (grandmother) 
will host at her house. Her son Robert Johnson and daughter-in-law Susan Johnson will 
bring the turkey. Robert and Susan's children, Tommy and Lisa, are excited to see their 
cousins.

Margaret's daughter Patricia and her husband Michael will drive up from Boston with 
their kids, Jake and Sophie. Patricia offered to make her famous apple pie, which 
Margaret praised as "even better than my own recipe."

Robert asked if Tommy could sit next to his cousin Jake at the kids' table, since 
they haven't seen each other since summer.
""".strip(),
        question="Who is Tommy's grandmother?",
        answer="Margaret Johnson",
        requires_entity_distinction=True,
        difficulty="medium",
        category="family"
    ))
    
    cases.append(CorefTestCase(
        id="family_002",
        text="""
Estate Planning Notes

Client: William Harrison
Spouse: Elizabeth Harrison

Children:
- James Harrison (son, age 45) married to Catherine
- Mary Harrison-Smith (daughter, age 42) married to John Smith

Grandchildren:
- Through James: Oliver (12), Emma (9)
- Through Mary: Sophie Smith (7)

William wishes to leave the family home to Elizabeth. If Elizabeth predeceases him, 
the home goes to James. The vacation property should go to Mary.

Elizabeth specifically requested that her jewelry collection go to her granddaughter 
Sophie, noting that Sophie has always admired her vintage pieces.
""".strip(),
        question="Who will receive Elizabeth's jewelry collection?",
        answer="Sophie Smith",
        requires_entity_distinction=True,
        difficulty="medium",
        category="family"
    ))
    
    return cases


def generate_conversation_scenarios(rng: random.Random) -> List[CorefTestCase]:
    """Generate multi-party conversation scenarios."""
    
    cases = []
    
    cases.append(CorefTestCase(
        id="conv_001",
        text="""
Slack Thread: #project-alpha

@jennifer.lee: Has anyone reviewed the PR for the payment integration?

@marcus.wong: I looked at it yesterday. The logic looks good but I'm concerned about error handling.

@jennifer.lee: Thanks Marcus! @sarah.patel can you add more error handling?

@sarah.patel: Sure, I'll update it today. Marcus, can you re-review once I push the changes?

@marcus.wong: Absolutely. Tag me when it's ready.

@jennifer.lee: Great teamwork! Sarah, make sure to add unit tests too.

@sarah.patel: Will do. I'll have everything ready by 3 PM.
""".strip(),
        question="Who wrote the original PR that needs more error handling?",
        answer="Sarah Patel (sarah.patel)",
        requires_entity_distinction=True,
        difficulty="medium",
        category="conversation"
    ))
    
    cases.append(CorefTestCase(
        id="conv_002",
        text="""
Email Thread: Re: Q3 Budget Approval

From: Amanda Foster <amanda.foster@company.com>
To: Budget Committee
Subject: Re: Q3 Budget Approval

Thanks for the feedback, everyone.

Brian's suggestion to reduce the marketing spend by 10% makes sense given our Q2 performance.

However, I disagree with Christina's proposal to cut the engineering budget. Our roadmap 
depends on those hires.

Derek, you mentioned a compromise - can you elaborate? I'd like to hear your thoughts 
before our Thursday meeting.

Amanda

---
Previous messages:
- Brian Chen proposed marketing cuts
- Christina Davis proposed engineering cuts  
- Derek Evans suggested a hybrid approach
""".strip(),
        question="Who proposed cutting the engineering budget?",
        answer="Christina Davis",
        requires_entity_distinction=True,
        difficulty="easy",
        category="conversation"
    ))
    
    return cases


def generate_multi_doc_scenarios(rng: random.Random) -> List[Tuple[CorefTestCase, CorefTestCase]]:
    """Generate scenarios requiring cross-document reasoning."""
    
    scenarios = []
    
    # Scenario 1: Customer support across tickets
    doc1 = CorefTestCase(
        id="multi_001a",
        text="""
Support Ticket #4521
Customer: Nathan Brooks
Email: nathan.brooks@email.com
Issue: Cannot reset password

Customer called in frustrated. He's been trying to reset his password for 3 days.
I escalated to Tier 2 support. Agent Olivia will follow up within 24 hours.
""".strip(),
        question="What is the customer's email address?",
        answer="nathan.brooks@email.com",
        requires_entity_distinction=True,
        difficulty="easy",
        category="multi_doc"
    )
    
    doc2 = CorefTestCase(
        id="multi_001b",
        text="""
Support Ticket #4522
Customer: Nathan Brooks  
Email: nathan.brooks@email.com
Issue: Follow-up on password reset

Olivia from Tier 2 resolved the issue. The customer's account was locked due to 
too many failed attempts. Olivia reset the lockout and the customer confirmed 
he can now access his account.

Customer expressed gratitude for Olivia's help and asked for her employee ID 
to submit positive feedback.
""".strip(),
        question="Who resolved Nathan's issue?",
        answer="Olivia",
        requires_entity_distinction=True,
        difficulty="easy",
        category="multi_doc"
    )
    
    scenarios.append((doc1, doc2))
    
    # Scenario 2: Project handoff
    doc3 = CorefTestCase(
        id="multi_002a",
        text="""
Project Handoff Notes - Phase 1

Outgoing lead: Peter Zhang
Incoming lead: Rachel Kim

Peter completed the initial architecture design and database schema.
Key decisions documented in Confluence. Peter recommends Rachel review 
the caching strategy first, as it's the most complex component.
""".strip(),
        question="Who designed the initial architecture?",
        answer="Peter Zhang",
        requires_entity_distinction=True,
        difficulty="easy",
        category="multi_doc"
    )
    
    doc4 = CorefTestCase(
        id="multi_002b",
        text="""
Project Status Update - Phase 2

Lead: Rachel Kim

After reviewing Peter's caching design, I've identified some optimization 
opportunities. I've documented my proposed changes and will discuss with 
Peter in next week's architecture review.

Current team: Rachel (lead), Sam (backend), Tina (frontend)
""".strip(),
        question="Who identified optimization opportunities in the caching design?",
        answer="Rachel Kim",
        requires_entity_distinction=True,
        difficulty="medium",
        category="multi_doc"
    )
    
    scenarios.append((doc3, doc4))
    
    return scenarios


def generate_challenge_set(seed: int = 42, n_cases: int = 50) -> Dict:
    """Generate the full coreference challenge set."""
    
    rng = random.Random(seed)
    
    all_cases = []
    
    # Generate scenarios
    all_cases.extend(generate_team_scenarios(rng))
    all_cases.extend(generate_family_scenarios(rng))
    all_cases.extend(generate_conversation_scenarios(rng))
    
    # Multi-doc scenarios
    multi_doc = generate_multi_doc_scenarios(rng)
    for doc1, doc2 in multi_doc:
        all_cases.append(doc1)
        all_cases.append(doc2)
    
    return {
        "metadata": {
            "name": "Coreference Challenge Set",
            "version": "1.0",
            "seed": seed,
            "total_cases": len(all_cases),
            "description": "Test cases where generic tags fail but semantic surrogates succeed",
            "categories": ["team", "family", "conversation", "multi_doc"],
            "purpose": "Demonstrate entity distinction advantage of semantic masking"
        },
        "cases": [asdict(c) for c in all_cases]
    }


def save_challenge_set(output_dir: str = ".", seed: int = 42):
    """Save the challenge set to JSON."""
    
    dataset = generate_challenge_set(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "coref_challenge_set.json")
    
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✓ Generated {dataset['metadata']['total_cases']} coreference challenge cases")
    print(f"✓ Saved to: {filepath}")
    
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Coreference Challenge Set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="../data")
    args = parser.parse_args()
    
    save_challenge_set(args.output, args.seed)
