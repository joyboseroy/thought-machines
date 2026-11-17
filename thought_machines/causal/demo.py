from .intervene import (average_effect, backdoor, confounded_net, confounded_net_with_effect,
                        intervene, observe, rain_net)


def main():
    net = confounded_net()
    print("== Ice cream and drowning (no causal link) ==")
    print(f"P(drowning | ice cream sold)      = {observe(net, 'Drowning', 'IceCream'):.3f}")
    print(f"P(drowning)                       = {net.prob({'Drowning': True}):.3f}")
    print(f"P(drowning | do(ice cream sold))  = {intervene(net, 'Drowning', 'IceCream'):.3f}")
    print(f"backdoor adjustment on weather    = {backdoor(net, 'Drowning', 'IceCream', ['Hot']):.3f}")

    net2 = confounded_net_with_effect()
    seen = observe(net2, "Drowning", "IceCream") - observe(net2, "Drowning", "IceCream", False)
    print("\n== Same confounder, small real effect ==")
    print(f"observed difference      {seen:+.3f}")
    print(f"causal effect, do()      {average_effect(net2, 'Drowning', 'IceCream'):+.3f}")
    adj = (backdoor(net2, 'Drowning', 'IceCream', ['Hot'])
           - backdoor(net2, 'Drowning', 'IceCream', ['Hot'], value=False))
    print(f"causal effect, backdoor  {adj:+.3f}")

    rain = rain_net()
    print("\n== Umbrellas and wet grass ==")
    print(f"P(wet grass | umbrella)       = {observe(rain, 'WetGrass', 'Umbrella'):.3f}")
    print(f"P(wet grass | do(umbrella))   = {intervene(rain, 'WetGrass', 'Umbrella'):.3f}")
    print(f"P(wet grass | do(rain))       = {intervene(rain, 'WetGrass', 'Rain'):.3f}")


if __name__ == "__main__":
    main()
