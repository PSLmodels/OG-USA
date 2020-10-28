library(devtools)
install_github("floswald/psidR")
library(openxlsx)
library(psidR)


# set directory in which to save data
script.dir <- getwd()
mydir <- file.path(script.dir)
print(paste0('Directory to save files = ', mydir))

# Read in PSID variable crosswalk file from PSID website
cwf <- read.xlsx("http://psidonline.isr.umich.edu/help/xyr/psid.xlsx")

# Set survey years to pull
years_to_pull = c(1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977,
                  1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
                  1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
                  1999, 2001, 2003, 2005, 2007, 2009, 2011, 2013, 2015, 2017)

# Get list of variables codes for each variable in each year
family_var_names <- list(# Family variables
                        interview_number="ER60002",
                        fam_smpl_wgt_core="V21547",
                        # Demographics
                        head_age="ER17013",
                        spouse_age="ER47319",
                        head_gender="ER47318",
                        head_num_children="V10977",
                        num_children="ER37724",
                        num_children_away_from_home="V561",
                        num_children_under18="ER47320",
                        # race
                        head_race="ER64810",
                        spouse_race="ER64671",
                        head_hispanic="ER64809",
                        spouse_hispanic="ER64670",
                        # married
                        head_marital_status="ER60024", # only 1977+
                        marital_status="V239", # all years
                        # education
                        head_yrs_educ="ER65459",
                        spouse_yrs_educ="ER65460",
                        # food stamps
                        value_foodstamps_lastsurveyyear="ER60720",
                        # AFDC/TANF
                        head_whether_receive_afdc_prior_year="ER61101",
                        spouse_whether_receive_afdc_prior_year="ER61505",
                        whether_receive_afdc_year_b4_last=" ER65077",
                        # supplemental security income
                        head_ssi_prior_year="ER65256",
                        spouse_ssi_prior_year="ER65286",
                        other_familyunit_ssi_prior_year="ER65286",
                        whether_receive_ssi_year_b4_last=" ER65078",
                        # other welfare
                        head_other_welfare_prior_year="ER65258",
                        spouse_other_welfare_prior_year="ER65288",
                        other_familyunit_other_welfare_prior_year="ER65326",
                        # unemp income
                        head_unemp_inc_prior_year="ER65270",
                        spouse_unemp_inc_prior_year="ER65300",
                        other_familyunit_unemp_inc_prior_year="ER65332",
                        # workers comp
                        head_workers_comp_prior_year="ER65272",
                        spouse_workers_comp_prior_year="ER65302",
                        other_familyunit_workers_comp_prior_year="ER65334",
                        # veterans pension
                        head_vet_pen_prior_year="ER65260",
                        spouse_vet_pen_prior_year="ER65290",
                        other_familyunit_vet_pen_prior_year="ER65328",
                        # federal income tax - get from PSID, but maybe compute from TaxSIM?
                        # Not in PSID - may be able to compute with the below variables
                        head_spouse_taxable_inc="ER65253",
                        other_familyunit_taxable_inc="ER65321",
                        head_spouse_tax_table="V19157",
                        head_spouse_num_tax_exemptions="V19156", # only available 1980-1992
                        # state income tax - - get from PSID, but maybe compute from TaxSIM?
                        # Not in PSID
                        # tax filing status
                        # Not in PSID
                        # total food expenditures
                        food_out_expend="V21711",
                        food_in_expend="ER60739",
                        # labor income
                        head_labor_inc="ER65216",
                        spouse_labor_inc_post1993="ER65244",
                        spouse_labor_inc_pre1993="V23324",
                        # number of non corp businesses
                        head_num_biz="ER65199",
                        spouse_num_biz="ER65227",
                        family_num_biz="ER65194",
                        # capital income
                        other_familyunit_asset_inc="ER65319",
                        head_dividend_inc="ER65219",
                        spouse_dividend_inc="ER65247",
                        head_interest_inc="ER65221",
                        spouse_interest_inc="ER65249",
                        head_rent_inc="ER65217",
                        spouse_rent_inc="ER65245",
                        # total income
                        family_total_inc="ER65349",
                        # transfer income
                        head_and_spouse_transfer_income="ER65314",
                        other_familyunit_transfer_income="ER65342",
                        # social security income
                        head_socsec_income="ER65343",
                        spouse_socsec_income="ER65345",
                        other_familyunit_socsec_income="ER65347",
                        # income from businesses (Sch C, E)
                        head_noncorp_bus_asset_income="ER52217",
                        spouse_noncorp_bus_asset_income="ER52247",
                        head_noncorp_bus_labor_income="ER65197",
                        spouse_noncorp_bus_labor_income="ER65225",
                        noncorp_businc="ER52211",
                        # Labor supply
                        work_fam_bus="V21306",
                        head_emp_status="ER7163",
                        spouse_emp_status="ER7657",
                        head_year_retire="ER47451",
                        spouse_year_retire="ER47708",
                        # hours
                        head_annual_hours="ER65156",
                        spouse_annual_hours="ER65177",
                        # for work in noncorp biz, we only know if they reported
                        # this in total work hours, 1996+
                        head_whether_report_biz_hours1="ER60929",
                        head_whether_report_biz_hours2="ER60943",
                        head_whether_report_biz_hours3="ER60957",
                        spouse_whether_report_biz_hours1="ER60931",
                        spouse_whether_report_biz_hours2="ER60945",
                        spouse_whether_report_biz_hours3="ER60959",
                        # Family wealth variables
                        net_wealth1="ER65406",
                        net_wealth2="ER65408",
                        other_debts="ER43611",
                        other_assets="ER48931",
                        net_real_estate_not_primary="ER52354",
                        mortgage1="ER60049",
                        mortgage2="ER60070",
                        house_value="ER60031",
                        net_vehicle_wealth="ER61731",
                        IRA_annuities="ER65376",
                        stocks="ER65368",
                        checking_savings="ER65368",
                        business_assets="ER52346",
                        whether_receive_inheritance="ER49041",
                        whether_receive_inheritance_1st="ER49041",
                        whether_receive_inheritance_2nd="ER49046",
                        whether_receive_inheritance_3rd="ER49051",
                        year_receive_inheritance_1st="ER43697",
                        year_receive_inheritance_2nd="ER43702",
                        year_receive_inheritance_3rd="ER43707",
                        inheritance="ER43523",
                        # 4th only available in one year, 1994
                        whether_receive_inheritance_4th="ER3851",
                        # Number of receipts of inheritance only available in 1984 and 1989
                        number_receive_inheritance="V17382",
                        value_inheritance_1st="ER49043",
                        value_inheritance_2nd="ER49048",
                        value_inheritance_3rd="ER49053"
                        ## There are variables on parents age/death, parent's wealth (only in 1988), parents education
                        ## skipping this for now, but maybe interesting to explore timing of bequests and death of parents
                      )

# Individual variables
ind_var_names <- list(ind_hours="ER30823")

# Create dataframes of variable names for each year
famvars <- data.frame(year=years_to_pull)
for (var in names(family_var_names)){
  famvars[[var]] <- c(getNamesPSID(family_var_names[[var]], cwf, years=years_to_pull))
}
indvars <- data.frame(year=years_to_pull)
for (var in names(ind_var_names)){
  indvars[[var]] <- c(getNamesPSID(ind_var_names[[var]], cwf, years=years_to_pull))
}
print('Beginning to build panel')
# Build PSID panel
psid_df <- build.panel(datadir=mydir, fam.vars=famvars, ind.vars=indvars, design='all')
save(psid_df,file=file.path(script.dir, 'psid_data_files', 'psid1968to2017.RData'))
