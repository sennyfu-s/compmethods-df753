# install.packages(c("shiny", "ggplot2", "dplyr", "shinythemes"))
library(shiny)
library(ggplot2)
library(dplyr)
library(shinythemes)

ui <- fluidPage(
  theme = shinytheme("flatly"),
  titlePanel("MetaCare"),
  sidebarLayout(
    sidebarPanel(
      radioButtons(
        "role",
        "Who are you?",
        choices = c("Patient", "Doctor"),
        inline = TRUE),
#------Patient UI------
      conditionalPanel(
        condition = "input.role == 'Patient'",
        h4("Patient Data Entry"),
        dateInput("p_date", "Date", value = Sys.Date()),
        numericInput("p_sys", "Systolic BP (mmHg)", value = NA, min = 60, max = 250),
        numericInput("p_dia", "Diastolic BP (mmHg)", value = NA, min = 40, max = 140),
        numericInput("p_glu", "Blood Glucose (mg/dL)", value = NA, min = 50, max = 400),
        numericInput("p_weight", "Weight (kg)", value = NA, min = 30, max = 250),
        actionButton("add_record", "Add record"),
        hr(),
        h5("Threshold settings"),
        numericInput("th_sys", "High systolic BP threshold", value = 140, min = 80, max = 250),
        numericInput("th_glu", "High glucose threshold", value = 180, min = 70, max = 400),
        helpText("If your latest values exceed these thresholds, the app will show warning pop-ups
                 with lifestyle or medication reminders. Extremely high values will trigger a
                 conceptual 'call 911' confirmation.")),
#------Doctor UI------
      conditionalPanel(
        condition = "input.role == 'Doctor'",
        h4("Doctor View (FHIR mock)"),
        textInput("doc_patient_id", "Patient ID"),
        actionButton("doc_fetch", "Fetch recent data (FHIR mock)"),
        helpText("In a real implementation, this would call a FHIR endpoint to retrieve recent
                 Observations for the selected patient. Here we generate mock data."))),
    mainPanel(
#------patient's main panel------
      conditionalPanel(
        condition = "input.role == 'Patient'",
        h3("Patient Dashboard"),
        uiOutput("patient_alert"),
        tabsetPanel(
          tabPanel("Trends",
                   br(),
                   plotOutput("patient_bp_trend"),
                   plotOutput("patient_glu_trend")
          ),
          tabPanel("Raw Data",
                   br(),
                   tableOutput("patient_table")))),
#------Doctor's main panel------
      conditionalPanel(
        condition = "input.role == 'Doctor'",
        h3("Doctor Dashboard"),
        uiOutput("doc_summary"),
        tabsetPanel(
          tabPanel("Trends",
                   br(),
                   plotOutput("doc_bp_trend"),
                   plotOutput("doc_glu_trend")
          ),
          tabPanel("Raw Data",
                   br(),
                   tableOutput("doc_table")))))))
server <- function(input, output, session) {
  # Store patient data entered in the UI
  vals <- reactiveValues(
    patient_data = data.frame(
      date = as.Date(character()),
      systolic = numeric(),
      diastolic = numeric(),
      glucose = numeric(),
      weight = numeric()),
    doc_data = NULL)
  # Critical threshold
  critical_sys <- reactive(180)
  # Patient data entry
  observeEvent(input$add_record, {
    req(input$p_date)
    # Add new row
    new_row <- data.frame(
      date = as.Date(input$p_date),
      systolic = input$p_sys,
      diastolic = input$p_dia,
      glucose = input$p_glu,
      weight = input$p_weight)
    vals$patient_data <- bind_rows(vals$patient_data, new_row) |> arrange(date)
    # Check thresholds for pop-up logic
    latest <- tail(vals$patient_data, 1)
    sys_val <- latest$systolic
    glu_val <- latest$glucose
    sys_high <- sys_val >= input$th_sys
    sys_critical <- sys_val >= critical_sys()
    glu_high <- glu_val >= input$th_glu
    # 1) Extremely high systolic BP
    if (sys_critical) {
      showModal(
        modalDialog(
          title = "Blood Pressure Abnormally High",
          tagList(
            p(
              paste0("Your latest systolic blood pressure reading is ",
                     sys_val, " mmHg, which is abnormally high.")),
            p("If you are experiencing symptoms such as chest pain, severe shortness of breath, confusion, or weakness, this could be a medical emergency."),
            strong("This prototype cannot make clinical decisions, but in a real app this is where an emergency workflow would be triggered.")),
          footer = tagList(
            modalButton("Cancel"),
            actionButton("confirm_call_911", "Confirm to call 911")),
          easyClose = TRUE))
    } else if (sys_high || glu_high) {
      # 2) Slihtly above regular threshold(s)
      probs <- c()
      if (sys_high) probs <- c(probs, "blood pressure")
      if (glu_high) probs <- c(probs, "fasting glucose")
      showModal(
        modalDialog(
          title = "Values Higher Than Your Usual Range",
          tagList(
            p(
              paste(
                "Your latest reading is above the threshold for:",
                paste(probs, collapse = " and "), "."
              )
            ),
            tags$ul(
              if (sys_high) tags$li("Review your blood pressure plan (medications, sodium intake, stress management)."),
              if (glu_high) tags$li("Review recent meals and your glucose management plan.")
            ),
            p("Consider taking a short walk, practicing relaxation, or following the plan discussed with your clinician."),
            p("If you feel unwell or symptoms are severe, contact a healthcare professional or follow your emergency plan.")
          ),
          easyClose = TRUE,
          footer = modalButton("OK")))}})
  # "Confirm to call 911"
  observeEvent(input$confirm_call_911, {
    removeModal()
    showModal(
      modalDialog(
        title = "Emergency Action (Conceptual Demo)",
        p("In a real application, this step would initiate a call to emergency services "),
        p("For this prototype, please imagine the app now prompts the user to dial 911 or their local emergency number."),
        easyClose = TRUE,
        footer = modalButton("Close")))})
  # Patient alert page
  output$patient_alert <- renderUI({
    if (nrow(vals$patient_data) == 0) return(NULL)
    latest <- tail(vals$patient_data, 1)
    sys_high <- latest$systolic >= input$th_sys
    glu_high <- latest$glucose >= input$th_glu
    if (!sys_high && !glu_high) {
      div(
        class = "alert alert-success",
        "Good job! Keep up the healthy work!"
      )
    } else {
      msg <- "Your latest reading is above the threshold for: "
      probs <- c()
      if (sys_high) probs <- c(probs, "systolic blood pressure")
      if (glu_high) probs <- c(probs, "fasting glucose")
      
      div(
        class = "alert alert-warning",
        strong("Warning: "),
        paste0(msg, paste(probs, collapse = " and "), "."),
        p("You have also received a pop-up with lifestyle or medication reminders."))}})
  #Trend
  output$patient_bp_trend <- renderPlot({
    req(nrow(vals$patient_data) > 0)
    df <- vals$patient_data
    ggplot(df, aes(x = date)) +
      geom_line(aes(y = systolic, color = "Systolic")) +
      geom_point(aes(y = systolic, color = "Systolic")) +
      geom_line(aes(y = diastolic, color = "Diastolic")) +
      geom_point(aes(y = diastolic, color = "Diastolic")) +
      geom_hline(yintercept = input$th_sys, linetype = "dashed", color = "darkred", alpha = 0.7) +
      scale_color_manual(values = c("Systolic" = "red", "Diastolic" = "blue")) +
      labs(
        y = "Blood Pressure (mmHg)",
        x = "Date",
        title = "Blood Pressure Trend (Patient View)",
        color = "Measurement"
      ) +
      theme_minimal()})
  output$patient_glu_trend <- renderPlot({
    req(nrow(vals$patient_data) > 0)
    df <- vals$patient_data
    ggplot(df, aes(x = date)) +
      geom_line(aes(y = glucose)) +
      geom_point(aes(y = glucose)) +
      geom_hline(yintercept = input$th_glu, linetype = "dashed")+
      labs(
        y = "Fasting Glucose (mg/dL)",
        x = "Date",
        title = "Glucose Trend (Patient View)")+
      theme_minimal()})
  output$patient_table <- renderTable({
    req(vals$patient_data)
    df <- vals$patient_data
    df$date <- as.Date(df$date, origin = "1970-01-01")
    df$date <- format(df$date, "%Y-%m-%d")
    df})
  # Doctor FHIR connection page
  observeEvent(input$doc_fetch, {
    set.seed(123)
    dates <- seq(Sys.Date() - 14, Sys.Date(), by = "day")
    df <- data.frame(
      date = as.Date(dates),
      systolic = round(rnorm(length(dates), 130, 10)),
      diastolic = round(rnorm(length(dates), 80, 8)),
      glucose = round(rnorm(length(dates), 150, 20)),
      weight = round(rnorm(length(dates), 80, 3), 1))
    vals$doc_data <- df})
  output$doc_summary <- renderUI({
    req(vals$doc_data)
    div(
      class = "alert alert-info",
      paste("Showing last 2 weeks of data for patient ID:", input$doc_patient_id)
    )
  })
  # Doctor trend reading
  output$doc_bp_trend <- renderPlot({
    req(vals$doc_data)
    df <- vals$doc_data
    # Make sure date is Date class
    df$date <- as.Date(df$date, origin = "1970-01-01")
    ggplot(df, aes(x = date)) +
      geom_line(aes(y = systolic, color = "Systolic")) +
      geom_point(aes(y = systolic, color = "Systolic")) +
      geom_line(aes(y = diastolic, color = "Diastolic")) +
      geom_point(aes(y = diastolic, color = "Diastolic")) +
      scale_color_manual(values = c("Systolic" = "red", 
                                    "Diastolic" = "blue")) +
      labs(
        y = "Blood Pressure (mmHg)",
        x = "Date",
        color = "Measurement",
        title = paste("Blood Pressure Trend (Doctor View –", input$doc_patient_id, ")"))+
      theme_minimal()
  })
  output$doc_glu_trend <- renderPlot({
    req(vals$doc_data)
    df <- vals$doc_data
    df$date <- as.Date(df$date, origin = "2000-01-01")
    ggplot(df, aes(x = date))+
      geom_line(aes(y = glucose))+
      geom_point(aes(y = glucose))+
      labs(
        y = "Glucose (mg/dL)",
        x = "Date",
        title = paste("Glucose Trend (Doctor View –", input$doc_patient_id, ")"))+
      theme_minimal()})
  output$doc_table <- renderTable({
    req(vals$doc_data)
    df <- vals$doc_data
    df$date <- as.Date(df$date, origin = "2000-01-01")
    df$date <- format(df$date, "%Y-%m-%d")
    df})}
# Call shiny
shinyApp(ui, server)

