import os

import banana_dev as banana
import dotenv

dotenv.load_dotenv(".env")

api_key = os.environ.get("BANANA_DEV_API_KEY", None)
model_key = os.environ.get("MODEL_KEY", None)

payload = {
    "context": """Land Rover is a British brand of predominantly four-wheel drive, off-road capable vehicles, owned by multinational car manufacturer Jaguar Land Rover (JLR), since 2008 a subsidiary of India's Tata Motors.[3] JLR currently builds Land Rovers in Brazil, China, India, Slovakia, and the United Kingdom. The Land Rover name was created in 1948 by the Rover Company for a utilitarian 4WD off-road vehicle; currently, the Land Rover range comprises solely of upmarket and luxury sport utility vehicles.

Land Rover was granted a Royal Warrant by King George VI in 1951,[4][5] and 50 years later, in 2001, it received a Queen's Award for Enterprise for outstanding contribution to international trade.[6] Over time, Land Rover grew into its own brand (and for a while also a company), encompassing a consistently growing range of four-wheel drive, off-road capable models. Starting with the much more upmarket 1970 Range Rover, and subsequent introductions of the mid-range Discovery and entry-level Freelander line (in 1989 and 1997), as well as the 1990 Land Rover Defender refresh, the marque today includes two models of Discovery, four distinct models of Range Rover, and after a three-year hiatus, a second generation of Defenders have gone into production for the 2020 model year—in short or long wheelbase, as before.

For half a century (from the original 1948 model, through 1997, when the Freelander was introduced), Land Rovers and Range Rovers exclusively relied on their trademark boxed-section vehicle frames. Land Rover used boxed frames in a direct product bloodline until the termination of the original Defender in 2016; and their last body-on-frame model was replaced by a monocoque with the third generation Discovery in 2017. Since then all Land Rovers and Range Rovers have a unified body and frame structure.

Since 2010, Land Rover has also introduced two-wheel drive variants, both of the Freelander, and of the Evoque, after having built exclusively 4WD cars for 62 years.[7] The 2WD Freelander has been succeeded by a 2WD Discovery Sport, available in some markets.[8]
""",
    "question": "What are Land Rover line-up products?"
}

out = banana.run(api_key, model_key, payload)
print(out["modelOutputs"][0])
