"""Natural controlled vocabulary shared by grounded artifact factories."""

FIRST_NAMES = (
    "Asha", "Mira", "Theo", "Devin", "Leila", "Omar", "Priya", "Jordan",
    "Elena", "Marcus", "Sofia", "Daniel", "Grace", "Maya", "Nolan", "Harper",
    "Isaiah", "Camille", "Rowan", "Nadia", "Lena", "Caleb", "Rosa", "Evan",
    "Talia", "Jonah", "Amara", "Miles", "Inez", "Simon", "Fatima", "Lucas",
)
LAST_NAMES = (
    "Bennett", "Carter", "Diaz", "Ellis", "Foster", "Gupta", "Hale", "Ibrahim",
    "Jensen", "Kim", "Lawson", "Morris", "Nolan", "Owens", "Patel", "Quinn",
    "Reed", "Santos", "Turner", "Usman", "Vega", "Walker", "Xu", "Young",
    "Zimmer", "Brooks", "Chen", "Davis", "Evans", "Flores", "Green", "Howard",
)
CITIES = (
    "Riverton", "Harborview", "Cedar Grove", "Oakridge", "Lakewood", "Fairmont",
    "Brookfield", "Pinehurst", "Maple Falls", "Stonebridge", "Westfield", "Clearwater",
)
VENUES = (
    "Civic Center", "Riverside Park", "Market Square", "Arts Pavilion",
    "Convention Hall", "Waterfront Plaza", "Community Green", "Town Commons",
)
ORGANIZATION_NAMES = (
    "Northwind Labs", "Harborview Museum", "Maple Street Market", "Cedar Grove Clinic",
    "Summit Bookshop", "Riverside Theater", "Oakridge Community Center", "Bluebird Transit",
    "Fairmont Arts Council", "Clearwater Foods", "Stonebridge Housing Group", "Pinehurst School Board",
)
COMPANY_NAMES = (
    "Northwind Systems", "Cedar Arc Technologies", "Blue Harbor Robotics", "Summit Ridge Energy",
    "Maple Cloud Software", "Clearwater Devices", "Stonebridge Mobility", "Pinecrest Analytics",
    "Lakewood Health Systems", "Fairmont Logistics", "Riverton BioLabs", "Harborlight Networks",
)
PROJECT_NAMES = (
    "Atlas", "Beacon", "Cobalt", "Drift", "Evergreen", "Falcon", "Horizon", "Juniper",
    "Keystone", "Lighthouse", "Meridian", "Northstar", "Orchard", "Pioneer", "Quarry", "Solstice",
)


def full_name(index: int) -> str:
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"
