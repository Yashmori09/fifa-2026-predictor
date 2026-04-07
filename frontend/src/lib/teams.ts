export interface Team {
  name: string;
  code: string; // ISO 3166-1 alpha-2 for flag-icons
  group: string;
  elo?: number;
}

// Actual FIFA 2026 World Cup draw — 48 teams, 12 groups
export const TEAMS: Team[] = [
  // Group A
  { name: "Mexico", code: "mx", group: "A" },
  { name: "South Korea", code: "kr", group: "A" },
  { name: "Czech Republic", code: "cz", group: "A" },
  { name: "South Africa", code: "za", group: "A" },
  // Group B
  { name: "Canada", code: "ca", group: "B" },
  { name: "Bosnia and Herzegovina", code: "ba", group: "B" },
  { name: "Qatar", code: "qa", group: "B" },
  { name: "Switzerland", code: "ch", group: "B" },
  // Group C
  { name: "Brazil", code: "br", group: "C" },
  { name: "Morocco", code: "ma", group: "C" },
  { name: "Haiti", code: "ht", group: "C" },
  { name: "Scotland", code: "gb-sct", group: "C" },
  // Group D
  { name: "United States", code: "us", group: "D" },
  { name: "Paraguay", code: "py", group: "D" },
  { name: "Australia", code: "au", group: "D" },
  { name: "Turkey", code: "tr", group: "D" },
  // Group E
  { name: "Germany", code: "de", group: "E" },
  { name: "Curaçao", code: "cw", group: "E" },
  { name: "Ivory Coast", code: "ci", group: "E" },
  { name: "Ecuador", code: "ec", group: "E" },
  // Group F
  { name: "Netherlands", code: "nl", group: "F" },
  { name: "Japan", code: "jp", group: "F" },
  { name: "Sweden", code: "se", group: "F" },
  { name: "Tunisia", code: "tn", group: "F" },
  // Group G
  { name: "Belgium", code: "be", group: "G" },
  { name: "Egypt", code: "eg", group: "G" },
  { name: "Iran", code: "ir", group: "G" },
  { name: "New Zealand", code: "nz", group: "G" },
  // Group H
  { name: "Spain", code: "es", group: "H" },
  { name: "Cape Verde", code: "cv", group: "H" },
  { name: "Saudi Arabia", code: "sa", group: "H" },
  { name: "Uruguay", code: "uy", group: "H" },
  // Group I
  { name: "France", code: "fr", group: "I" },
  { name: "Senegal", code: "sn", group: "I" },
  { name: "Iraq", code: "iq", group: "I" },
  { name: "Norway", code: "no", group: "I" },
  // Group J
  { name: "Argentina", code: "ar", group: "J" },
  { name: "Algeria", code: "dz", group: "J" },
  { name: "Austria", code: "at", group: "J" },
  { name: "Jordan", code: "jo", group: "J" },
  // Group K
  { name: "Portugal", code: "pt", group: "K" },
  { name: "DR Congo", code: "cd", group: "K" },
  { name: "Uzbekistan", code: "uz", group: "K" },
  { name: "Colombia", code: "co", group: "K" },
  // Group L
  { name: "England", code: "gb-eng", group: "L" },
  { name: "Croatia", code: "hr", group: "L" },
  { name: "Ghana", code: "gh", group: "L" },
  { name: "Panama", code: "pa", group: "L" },
];

export const GROUPS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
];

export function getTeamByName(name: string): Team | undefined {
  return TEAMS.find((t) => t.name === name);
}

export function getTeamsByGroup(group: string): Team[] {
  return TEAMS.filter((t) => t.group === group);
}
