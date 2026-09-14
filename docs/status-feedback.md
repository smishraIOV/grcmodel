# Observations from  docs/status-report.md

## Horizon and decision fequency

I am trying to think of a balance between young startup and mature firm, while still having a meaningful GRC model.

How about we try a horizon of 5 years. And decision frequency of 2 weeks - to match the sprint planning and review duration common in tech (including fintech firms)

## Section 1
L43: is "GRC function" the same thing as a GRC program?

L47: depositors leaving "smoothly" instead of suddenly. I don't follow. Isn't a run kind of a panic situation. Let's rephrase to match what model intent is here

## Section 2

L56: two periods. this is motivated from Froot Stein's model. Let's add that.
L57: a "random loss"? not sure what this is. I thought we had random investment opportunitites. clarify the phrasing to match what the model does or did

## section 3
L83: what is the 25%/year figure?

L85: cannot deploy more than it can fund. Ok. How is this constraint determined? 

L86 says the firm can fail three ways. but the bullet points underneath make it seem like there are more than three. A rewrite of this section to clarify.

## section 4
subsections 4.1, 4.2, 4.3 can be removed.

L131: let's not say rejected. it remains a branch, the svg branch as an experiment. if we discover instabilities in the future, we could explore this path further

subsections 4.5 and 4.6 can also be removed

subsection 4.7 most of this can be removed. A few important bits that are stll in the model can be considered to be merged in section 6 

subsection 4.8 : this is related to L127. Most of this subsection can be eliminated. Keep only a brief summary. Clarify whether this constant policy is taken as given, or if the constant level if optimized.
Then extract the text to its own section

subsection 4.9: can get rid of this. As mentioned at the top of this file I am considering modifying the horizon to 5 years with biweekly decisions.

## Section 5
The main point here is that the parameters are not calibrated. To me, this means none of the takeaways are credible. We need to think about methods to handle calibration - in a gruadual but serious manner. We can brainstorm ways to do this.

## Section 6
L340. This assumption is an example of something that is not very credible.

L351: the learner has become unstable. Along with the above lack of credibility and robustness, does this not provide some support to keeping the svg experiment in good shape, even if it performance is currently not that great?

L356: there's room for improvement regarding the continuation value. Maybe expanding the horizon to 5 years can help a bit here? L380 again mentioned the limitations of the horizon.

