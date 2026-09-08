"""Instructor-side: download the licence-clean photos used in the MP4 lab from Wikimedia Commons,
resize them and write data/photos/<set>/credits.json with attribution. Run once:
    python build/curate_photos.py --out data/photos
Plan drawings (plan_*) come from the course's own repository (Haolan-Zhang/SAM_Example_Image).
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path
from PIL import Image

C = "https://upload.wikimedia.org/wikipedia/commons/"
G = "https://raw.githubusercontent.com/Haolan-Zhang/SAM_Example_Image/master/"

PHOTOS = [
    # ------------------------------------------------------------------ workshop: structural / site
    ("site", "site_01", "Formwork and rebar before a concrete pour, East Approach, New York, 2017", "MTA Capital Construction Mega Projects", "CC BY 2.0",
     C + "6/67/Formwork_and_rebar_in_advance_of_a_concrete_pour_at_the_East_Approach_for_the_planned_Amtrak_Westbound_Bypass_Tunnel_which_will_send_trains_below_the_existing_LIRR_tracks._%28CH057A%2C_9-13-2017%29_%2836406383744%29.jpg", "formwork, rebar"),
    ("site", "site_02", "Setting a rebar cage for a foundation, Queens, 2018", "MTA Capital Construction Mega Projects", "CC BY 2.0",
     C + "9/94/Setting_a_rebar_cage_for_a_foundation_east_of_Queens_Blvd._%28CQ033%2C_10-17-2018%29_%2845378839502%29.jpg", "rebar cage, crane"),
    ("site", "site_03", "Rebar on a building site, Hölzla", "Ermell", "CC BY-SA 4.0", C + "7/7c/Baustelle-H%C3%B6lzla-Rebar-4141399.jpg", "rebar"),
    ("site", "site_04", "Rebar on a building site, Hölzla (2)", "Ermell", "CC BY-SA 4.0", C + "3/3b/Baustelle-H%C3%B6lzla-Rebar-6228085.jpg", "rebar"),
    ("site", "site_05", "Rebar for bridge foundation piling", "NPS Photo", "Public domain",
     C + "4/43/Rebar_for_Bridge_Foundation_Piling_%2875e8f25b-9308-4e22-9ab4-fdf4bb4adaf2%29.JPG", "rebar cage"),
    ("site", "site_06", "Builders tie a rebar cage together", "U.S. Navy", "Public domain",
     C + "0/07/US_Navy_080929-N-8816D-009_Builder_3rd_Class_Christopher_Dutra_and_Utilitiesman_3rd_Class_Thomas_E._Hanson_tie_a_rebar_cage_together.jpg", "rebar cage, workers"),
    ("site", "site_07", "Pouring concrete footings, Trimingham, 8 Jan 2021", "Kolforn", "CC BY-SA 4.0",
     C + "5/56/-2021-01-08_Men_pouring_concrete_footings%2C_Trimingham%2C_Norfolk.JPG", "series A day 1: concrete footings being poured"),
    ("site", "site_08", "Brick and block footings, Trimingham, 13 Jan 2021", "Kolforn", "CC BY-SA 4.0",
     C + "6/62/-2021-01-13_Brick_and_block_footings%2C_Trimingham%2C_Norfolk_%281%29.JPG", "series A day 6: blockwork on the footings"),
    ("site", "site_09", "Foundations and concrete oversite, Trimingham, 18 Jan 2021", "Kolforn", "CC BY-SA 4.0",
     C + "5/51/-2021-01-18_Foundations_and_concrete_oversite%2C_Trimingham%2C_Norfolk_%281%29.JPG", "series A day 11: concrete oversite poured"),
    ("site", "site_10", "Foundations and concrete oversite, Trimingham, 18 Jan 2021 (3)", "Kolforn", "CC BY-SA 4.0",
     C + "5/57/-2021-01-18_Foundations_and_concrete_oversite%2C_Trimingham%2C_Norfolk_%283%29.JPG", "series A day 11, other view"),
    ("site", "site_11", "Concrete pouring for a new school, Spangdahlem", "U.S. Air Force / Airman Sydney Franklin", "Public domain",
     C + "0/0e/Concrete_pouring_for_the_new_Spangdahlem_Elementary_School_%288062807%29.jpg", "concrete pour, workers"),
    ("site", "site_12", "Concrete pouring for a new school, Spangdahlem (2)", "U.S. Air Force / Airman Sydney Franklin", "Public domain",
     C + "8/87/Concrete_pouring_for_the_new_Spangdahlem_Elementary_School_%288062808%29.jpg", "concrete pour, workers"),
    ("site", "site_13", "Construction site with a concrete pump truck", "Steve Pivnick, U.S. Air Force", "Public domain",
     C + "4/4c/Construction_site_with_concrete_pump_truck.JPG", "concrete pump"),
    ("site", "site_14", "Seabees pour concrete at a project site, Senegal", "U.S. Navy / MC1 Ryan G. Wilber", "Public domain",
     C + "c/c5/US_Navy_110520-N-AW868-082_Seabees_assigned_to_Naval_Mobile_Construction_Battalion_%28NMCB%29_74%2C_Detachment_Senegal%2C_pour_concrete_at_a_project_site_a.jpg", "concrete pour, formwork"),
    ("site", "site_15", "Workers pouring concrete from a boom pump", "Grendelkhan", "CC BY-SA 3.0",
     C + "1/19/Workers_pouring_concrete_from_a_boom_pump.jpg", "concrete pour"),
    ("site", "site_16", "Construction of a new supermarket, Biddulph, 2010", "Jonathan Kington", "CC BY-SA 2.0",
     C + "3/34/-2010-09-10_Construction_of_New_Sainsbury%E2%80%99s_supermarket%2C_Biddulph%2C_Staffordshire.jpg", "steel frame, site"),
    ("site", "site_17", "Scaffolding on a new education building block", "State Government Photographer (Queensland)", "CC0",
     C + "7/7c/Scaffolding_on_new_education_building_block%2C_under_construction_-_Contractor-_J._Brown%28GN11799%29.jpg", "scaffolding"),
    ("site", "site_18", "Constructions and scaffolding at the Oosterdokseiland, Amsterdam", "Fons Heijnsbroek", "CC0",
     C + "7/79/Constructions_and_scaffolding_at_the_Oosterdokseiland%2C_Amsterdam-Centrum.jpg", "scaffolding, concrete frame"),
    ("site", "site_19", "A building frame with scaffolding at a construction site", "Shixart1985", "CC BY 2.0",
     C + "3/3c/A_building_frame_with_scaffolding_at_a_construction_site_with_blue_sky_in_background.jpg", "scaffolding, frame"),
    ("site", "site_20", "The Milliners apartment building under construction, September 2025", "Captain Galaxy", "CC BY 4.0",
     C + "d/d5/The_Milliners_apartment_building_under_construction_-_September_2025.jpg", "concrete frame, crane"),
    ("site", "site_21", "High-rise construction with scaffolding and protective netting, Makati", "Marek Ślusarczyk", "CC BY 3.0",
     C + "1/1a/11_Manila_Makati_-_high_rise_construction_scaffolding_and_protective_netting.jpg", "scaffolding, netting"),
    ("site", "site_22", "Hay Street parking deck under construction", "jalexartis", "CC BY 2.0", C + "3/3c/Hay_Street_Parking_Deck_82.jpg", "concrete deck"),
    # Belchertown 1937-1939: one building photographed through the whole project (public domain, DPLA)
    ("site", "site_23", "Belchertown administration building: foundation, 30 Aug 1937", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "b/bd/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_foundation_for_main_building%2C_Belchertown%2C_Mass.%2C_Aug._30%2C_1937_-_DPLA_-_126d5ba09bd2d1c414908ad76b64bed7.jpg", "series B 1: foundation excavation"),
    ("site", "site_24", "Belchertown: forms and reinforcing for the foundation, 10 Sep 1937", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "4/42/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_forms_and_reinforcing_for_administration_building_foundation%2C_Belchertown%2C_Mass.%2C_Sep._10%2C_1937_-_DPLA_-_d58279d71e9b65bc5f9295a0fdc44151.jpg", "series B 2: formwork and rebar"),
    ("site", "site_25", "Belchertown: foundation, 11 Oct 1937", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "3/3f/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_looking_southerly_at_foundation_for_administration_building%2C_Belchertown%2C_Mass.%2C_Oct._11%2C_1937_-_DPLA_-_626b8177a92e3f5680f0c7ea34a92c81.jpg", "series B 3: concrete foundation walls"),
    ("site", "site_26", "Belchertown: general view, 2 Nov 1937", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "2/25/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_general_view_of_administration_building%2C_looking_southerly%2C_Belchertown%2C_Mass.%2C_Nov._2%2C_1937_-_DPLA_-_c68db5273969b21025a6a117e7526b47.jpg", "series B 4"),
    ("site", "site_27", "Belchertown: steel for the main building, 30 Nov 1937", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "e/e1/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_steel_for_main_building%2C_Belchertown%2C_Mass.%2C_Nov._30%2C_1937_-_DPLA_-_dc4e81029ec11a601679993fba9999ab.jpg", "series B 5: steel frame"),
    ("site", "site_28", "Belchertown: laying brick, 20 Dec 1937", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "4/4a/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_laying_brick_on_north_wall_of_second_floor_of_the_administration_building%2C_Belchertown%2C_Mass.%2C_Dec._20%2C_1937_-_DPLA_-_c00980a59b473b66ab715104f6b099cd.jpg", "series B 6: brickwork"),
    ("site", "site_29", "Belchertown: looking south-east at the main building, 27 Jan 1938", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "6/6d/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_looking_southeast_at_main_building%2C_Belchertown%2C_Mass.%2C_Jan._27%2C_1938_-_DPLA_-_257591addc10b53b5a509e170b0e11ad.jpg", "series B 7"),
    ("site", "site_30", "Belchertown: main building, 12 May 1938", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "4/40/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_looking_northerly_at_Main_Building%2C_Belchertown%2C_Mass.%2C_May_12%2C_1938_-_DPLA_-_b8338b3f2860757149eb9036744b4a57.jpg", "series B 8"),
    ("site", "site_31", "Belchertown: administration buildings, 10 Aug 1938", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "3/3e/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_looking_westerly_at_administration_buildings%2C_Belchertown%2C_Mass.%2C_Aug.10%2C_1938_-_DPLA_-_7546b18738eb6ca10d5dff0b37e9e4ed.jpg", "series B 9"),
    ("site", "site_32", "Belchertown: finished administration building, 14 Sep 1939", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "5/5e/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_looking_southerly_at_Administration_Building%2C_Belchertown%2C_Mass.%2C_Sep._14%2C_1939_-_DPLA_-_24f996e7120a86174147c1f30567e15b.jpg", "series B 10: finished"),
    # ------------------------------------------------------------------ homework: interior / finishing
    ("interior", "int_01", "Drywall crane and workers", "NIOSH", "Public domain", C + "d/d1/Drywall_Crane_Workers.jpg", "drywall panels being installed"),
    ("interior", "int_02", "Drywall installation", "Loadmaster (David R. Tribble)", "CC BY-SA 3.0", C + "3/3f/Drywall-4221R.jpg", "drywall, studs"),
    ("interior", "int_03", "Dry wall", "MrBeastRapper", "CC BY-SA 4.0", C + "5/5e/Dry_wall.jpg", "drywall"),
    ("interior", "int_04", "Drywall with an outlet", "MrBeastRapper", "CC BY-SA 4.0", C + "6/61/Drywall_WITH_AN_OUTLET_%21.jpg", "drywall"),
    ("interior", "int_05", "Drywall and tools", "Timothyjosephwood", "CC BY-SA 4.0", C + "f/f5/Drywall_and_tools.jpg", "drywall"),
    ("interior", "int_06", "Gypsum board interior wall, Elm Place", "Marina-Shehata", "CC0", C + "6/61/Elm-Place-Dry-Wall-Gypsum-Board-Interior-wall.jpg", "drywall"),
    ("interior", "int_07", "House ready for drywall (studs and insulation)", "Riverview Homes Inc", "CC BY-SA 3.0", C + "6/62/Pine_Grove_Homes_Ready_For_Drywall.jpg", "studs, insulation"),
    ("interior", "int_08", "Plasterboard with tapered edges", "Panamitsu", "CC BY-SA 4.0", C + "2/2c/Gib_board_aka_plasterboard_tapered_edges_01.jpg", "plasterboard"),
    ("interior", "int_09", "Drywall penetrations in an industrial building", "Achim Hering", "CC BY 3.0", C + "4/4c/Texaco_nanticoke_drywall_penetrations.jpg", "drywall, pipes"),
    ("interior", "int_10", "Drywall firestop detail", "Achim Hering", "Public domain", C + "3/3c/Drywall_firestop_problem4.jpg", "drywall, steel"),
    ("interior", "int_11", "Belchertown: boiler room, 14 Jan 1938", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "b/b3/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_boiler_room%2C_main_building%2C_Belchertown%2C_Mass.%2C_Jan._14%2C_1938_-_DPLA_-_2474faa3cb0b84d7eb3802ecc65868fc.jpg", "boilers, pipes"),
    ("interior", "int_12", "Belchertown: plumbing in the basement toilet, 10 Mar 1938", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "e/e8/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_plumbing_in_basement_toilet%2C_main_administration_building%2C_Belchertown%2C_Mass.%2C_Mar._10%2C_1938_-_DPLA_-_711b6912716efa071420b48aa914e072.jpg", "pipes, walls"),
    ("interior", "int_13", "Belchertown: tiling in the first-floor toilet, 10 Jun 1938", "Metropolitan District Water Supply Commission (S. D. Pike)", "Public domain",
     C + "4/43/Contract_No._56%2C_Administration_Buildings%2C_Main_Dam%2C_Belchertown%2C_tiling_in_first_floor_toilet%2C_main_building%2C_administration_buildings%2C_Belchertown%2C_Mass.%2C_Jun._10%2C_1938_-_DPLA_-_6ae65bb7afd31a33e2d8de918b327836.jpg", "tiles"),
    ("interior", "int_14", "Third-floor addition, Koett House, Covington", "w_lemay", "CC BY-SA 2.0",
     C + "2/2b/3rd_Floor_Addition%2C_Albert_B._Koett_House%2C_Russell_Street%2C_Mutter_Gottes%2C_Covington%2C_KY.jpg", "timber framing"),
    ("interior", "int_15", "Drywall shaft damage", "Achim Hering", "Public domain", C + "5/55/Drywall_shaft_damage.jpg", "damaged drywall"),
    ("interior", "int_16", "Hanging drywall", "Thomas Depippo (U.S. Navy)", "Public domain", C + "8/8a/Hanging_Drywall_%289020064%29.jpg", "drywall being hung on a stud wall"),
    ("interior", "int_17", "Carpenters installing sheetrock, LIRR Concourse, 2019", "MTA Capital Construction Mega Projects", "CC BY 2.0",
     C + "5/51/Carpenters_installing_sheetrock_in_the_back_of_house_area_in_the_future_LIRR_Concourse._10-30-2019_%2848997634792%29.jpg", "drywall, metal studs"),
    ("interior", "int_18", "Drywall installed in an office hallway, Canaan Valley", "U.S. Fish and Wildlife Service", "Public domain",
     C + "5/57/Canaan_Valley_National_Wildlife_Refuge_Drywall_installed_in_private_office_hallway_%2849204773016%29.jpg", "drywall hallway"),
    ("interior", "int_19", "Seabees renovate office spaces, Osan (1)", "U.S. Navy / PO1 Justin Rayburn", "Public domain",
     C + "f/f7/NMCB_4_Seabees_Rennovate_Office_Spaces%2C_Osan%2C_South_Korea_%289391388%29.jpg", "interior renovation"),
    ("interior", "int_20", "Seabees renovate office spaces, Osan (2)", "U.S. Navy / PO1 Justin Rayburn", "Public domain",
     C + "e/e1/NMCB_4_Seabees_Rennovate_Office_Spaces%2C_Osan%2C_South_Korea_%289391396%29.jpg", "interior renovation"),
    ("interior", "int_21", "A builder marks sheetrock", "U.S. Navy / MC3 Patrick W. Mullen III", "Public domain",
     C + "5/56/US_Navy_070911-N-8547M-027_Builder_3rd_Class_Tabitha_E._Ball%2C_of_Naval_Mobile_Construction_Battalion_%28NMCB%29_5%27s_air_detachment%2C_marks_sheetrock_during_a_construction_project_that_battalion_Seabees_volunteered_to_carry_out.jpg", "sheetrock"),
    ("interior", "int_22", "Builders cut sheetrock", "U.S. Navy / MC3 Patrick W. Mullen III", "Public domain",
     C + "6/6f/US_Navy_070911-N-8547M-044_Lt._Song_Hwang%2C_chaplain_of_Naval_Mobile_Construction_Battalion_%28NMCB%29_5%2C_left%2C_and_Builder_Constructionman_Robert_L._Dynda%2C_of_the_battalion%27s_air_detachment%2C_cut_sheetrock_during_a_constructio.jpg", "sheetrock"),
    ("interior", "int_23", "Piping insulation on ventilation pipes, LIRR Concourse, 2019", "MTA Capital Construction Mega Projects", "CC BY 2.0",
     C + "5/57/Piping_insulation_installed_on_ventilation_pipes_in_the_back_of_house_area_of_the_concourse._%28CM0914B%2C_03-14-2019%29_%2847387054911%29.jpg", "pipes, insulation"),
    ("interior", "int_24", "Seabees small-scale construction project, Jamaica, 2024 (1)", "U.S. Navy / PO2 Adriones Johnson", "Public domain",
     C + "e/ee/Seabees_conduct_small-scale_construction_project_in_Jamaica_as_part_of_Continuing_Promise_2024_%288527372%29.jpg", "construction"),
    ("interior", "int_25", "Seabees small-scale construction project, Jamaica, 2024 (2)", "U.S. Navy / PO2 Adriones Johnson", "Public domain",
     C + "c/c6/Seabees_conduct_small-scale_construction_project_in_Jamaica_as_part_of_Continuing_Promise_2024_%288527368%29.jpg", "construction"),
    # ------------------------------------------------------------------ plans (course's own images)
    ("plans", "plan_01", "Structural plan 1 (footings)", "CEM4644 course material", "course material", G + "test_fp.png", "plan drawing"),
    ("plans", "plan_02", "Structural plan 2 (footings, elevator shaft)", "CEM4644 course material", "course material", G + "test_fp_2.png", "plan drawing"),
]


UA = "CEM4644-course-material/1.0 (educational use; https://github.com/Haolan-Zhang/CEM4644)"


def thumb_url(url: str, width: int = 1280) -> str:
    """Wikimedia serves resized copies from its CDN; the originals endpoint rate-limits bulk downloads."""
    if "/wikipedia/commons/" not in url or "/thumb/" in url:
        return url
    prefix, rest = url.split("/wikipedia/commons/", 1)
    return f"{prefix}/wikipedia/commons/thumb/{rest}/{width}px-{rest.split('/')[-1]}"


def fetch(url: str, dest: Path, tries: int = 4, width: int = 1280):
    candidates = [thumb_url(url, width), url] if "commons" in url else [url]
    for k in range(tries):
        for u in candidates:
            try:
                req = urllib.request.Request(u, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 20000 and b"<html" in data[:2000].lower():
                    raise RuntimeError("got an HTML page instead of an image")
                dest.write_bytes(data)
                time.sleep(1.5)  # be polite to the CDN
                return True
            except Exception as e:
                print(f"   attempt {k + 1} on {'thumb' if 'thumb' in u else 'original'}: {str(e)[:90]}")
                time.sleep(5 * (k + 1))
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "photos"))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--raw", default=None, help="keep the originals here")
    a = ap.parse_args()
    out = Path(a.out)
    credits = {}
    for group, key, title, author, lic, url, note in PHOTOS:
        d = out / group
        d.mkdir(parents=True, exist_ok=True)
        ext = ".png" if group == "plans" else ".jpg"
        dst = d / f"{key}{ext}"
        raw = Path(a.raw) / (key + Path(url).suffix.lower().split("%")[0]) if a.raw else None
        if raw and not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not fetch(url, raw):
                print("FAILED", key); continue
        src = raw if raw else dst
        if not raw:
            if not fetch(url, dst):
                print("FAILED", key); continue
        im = Image.open(src)
        im = im.convert("RGBA").convert("RGB") if group == "plans" else im.convert("RGB")
        if max(im.size) > a.max_side:
            s = a.max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        if group == "plans":
            im.save(dst, "PNG", optimize=True)
        else:
            im.save(dst, "JPEG", quality=88, optimize=True)
        credits.setdefault(group, []).append({"id": key, "file": dst.name, "title": title, "author": author, "license": lic,
                                              "source": url, "note": note, "size": im.size})
        print(f"{key}: {im.size} {lic} ({dst.stat().st_size // 1024} KB)")
    for group, lst in credits.items():
        (out / group / "credits.json").write_text(json.dumps(lst, indent=2, ensure_ascii=False))
    print("done:", {g: len(l) for g, l in credits.items()})
